from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module

from einops import pack, repeat, unpack
from einops.layers.torch import Rearrange
from x_transformers import Encoder

from utils.my_residual_vector_quantize import VectorQuantize as VQ
from .st2_encoder import build_st2_v7_encoder, load_st2_encoder_checkpoint


class Tokenizer(Module):
    def __init__(
        self,
        *,
        dim=128,
        seq_length=500,
        patch_size=25,
        channels=12,
        latent_ratio=0.5,
        enc_depth=6,
        enc_heads=8,
        enc_dim_head=64,
        dec_depth=6,
        dec_heads=8,
        dec_dim_head=64,
        codebook_size=256,
        freeze_foundation_encoder=True,
        enc_kwargs: dict = dict(),
        dec_kwargs: dict = dict(),
        vq_kwargs: dict = dict(),
        encoder_checkpoint_path: str | None = None,
    ):
        super().__init__()
        self.seq_length = seq_length
        self.patch_size = patch_size
        self.dim = dim
        self.freeze_foundation_encoder = freeze_foundation_encoder

        self.dim_patch = channels * patch_size
        self.num_tokens = seq_length // patch_size
        self.num_latent_tokens = int(self.num_tokens * latent_ratio)

        self.latents = nn.Parameter(torch.zeros(self.num_latent_tokens, dim))
        self.mask_tokens = nn.Parameter(torch.zeros(self.num_tokens, dim))
        nn.init.normal_(self.latents, std=0.02)
        nn.init.normal_(self.mask_tokens, std=0.02)

        self.pos_emb = nn.Embedding(1024, dim)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

        self.encoder_checkpoint_path = self._resolve_encoder_checkpoint(encoder_checkpoint_path)
        self.foundation_encoder = build_st2_v7_encoder(
            seq_len=seq_length,
            patch_size=patch_size,
            num_leads=channels,
        )
        load_st2_encoder_checkpoint(self.foundation_encoder, self.encoder_checkpoint_path)
        for param in self.foundation_encoder.parameters():
            param.requires_grad = not self.freeze_foundation_encoder
        torch.cuda.empty_cache()

        self.sequence_to_tokens = nn.Sequential(
            self.foundation_encoder,
            Rearrange('b n e d -> b n (e d)'),
            nn.Linear(self.foundation_encoder.output_dim, dim),
        )

        self.encoder = Encoder(
            dim=dim,
            depth=enc_depth,
            heads=enc_heads,
            attn_dim_head=enc_dim_head,
            **enc_kwargs,
        )

        self.vq = VQ(
            dim=dim,
            codebook_dim=dim,
            codebook_size=codebook_size,
            codebook_diversity_loss_weight=0.3,
            **vq_kwargs,
        )

        self.decoder = Encoder(
            dim=dim,
            depth=dec_depth,
            heads=dec_heads,
            attn_dim_head=dec_dim_head,
            **dec_kwargs,
        )

        self.tokens_to_sequence = nn.Sequential(
            nn.Linear(dim, self.dim_patch),
            Rearrange('b n (c p) -> b c (n p)', p=patch_size),
        )

    def _resolve_encoder_checkpoint(self, checkpoint_path):
        project_root = Path(__file__).resolve().parents[1]
        candidates = []
        if checkpoint_path is not None:
            candidates.append(Path(checkpoint_path))
        candidates.extend(
            [
                project_root / 'ckpt' / 'pretrained_weight' / 'CoRe-ECG_mimic_encoder.pth',
            ]
        )
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        raise FileNotFoundError(
            'Could not locate the CoRe-ECG mimic pretrained encoder checkpoint. '
            + ', '.join(str(candidate) for candidate in candidates)
        )

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_foundation_encoder:
            self.foundation_encoder.eval()
        return self

    def encode_patch_tokens(self, sequence):
        batch = sequence.shape[0]
        tokens = self.sequence_to_tokens(sequence)
        positions = torch.arange(tokens.size(1), device=tokens.device)
        pos_emb = self.pos_emb(positions).unsqueeze(0).expand(batch, -1, -1)
        return tokens + pos_emb

    def encode_patch_tokens_and_latents(self, sequence, quantize=False):
        batch = sequence.shape[0]
        patch_tokens = self.encode_patch_tokens(sequence)
        latents = repeat(self.latents, 'l d -> b l d', b=batch)
        tokens, packed_shape = pack([patch_tokens, latents], 'b * d')
        tokens = self.encoder(tokens)
        patch_tokens, latents = unpack(tokens, packed_shape, 'b * d')

        if quantize:
            latents, _, _ = self.vq(latents)

        return patch_tokens, latents

    def encode_latents(self, sequence, quantize=False):
        _, latents = self.encode_patch_tokens_and_latents(sequence, quantize=quantize)
        return latents

    def encode_features(self, sequence, pooling='flatten', quantize=False):
        latents = self.encode_latents(sequence, quantize=quantize)

        if pooling == 'none':
            return latents
        if pooling == 'mean':
            return latents.mean(dim=1)
        if pooling == 'flatten':
            return latents.flatten(1)

        raise ValueError(f'Unsupported pooling mode: {pooling}')

    @torch.no_grad()
    def tokenize(self, sequence):
        return self.forward(sequence, return_codebook_ids=True)

    def codebook_ids_to_sequence(self, token_ids):
        codes = self.vq.get_output_from_indices(token_ids)
        return self.decode(codes)

    def decode(self, latents):
        batch = latents.shape[0]
        positions = torch.arange(self.num_tokens, device=latents.device)
        pos_emb = self.pos_emb(positions).unsqueeze(0).expand(batch, -1, -1)

        mask_tokens = repeat(pos_emb, 'b n d -> b n d', b=batch)
        tokens, mask_packed_shape = pack([mask_tokens, latents], 'b * d')

        tokens = self.decoder(tokens)
        tokens, _ = unpack(tokens, mask_packed_shape, 'b * d')
        recon = self.tokens_to_sequence(tokens)
        return recon

    def predict(self, tokens, ar_context):
        num_pred_tokens = self.num_tokens
        for _ in range(num_pred_tokens):
            context = tokens[:, -ar_context:, :]
            next_token = self.ar_predictor(context)[0][:, -1:]
            tokens = torch.cat([tokens, next_token], dim=1)

        pred_tokens = tokens[:, self.num_tokens:, :]
        return pred_tokens

    def forward(
        self,
        sequence,
        return_codebook_ids=False,
        return_recon_sequence=False,
        predict=False,
    ):
        orig_sequence = sequence
        latents = self.encode_latents(sequence)

        quantized, indices, _ = self.vq(latents)

        if return_codebook_ids:
            return indices

        recon_sequence = self.decode(quantized)[:, :, :orig_sequence.shape[2]]
        recon_loss = F.mse_loss(recon_sequence, orig_sequence)

        if predict:
            pred_sequence = self.tokens_to_sequence(quantized)
            return recon_loss, recon_sequence, pred_sequence

        if not return_recon_sequence:
            return recon_loss

        return recon_loss, recon_sequence
