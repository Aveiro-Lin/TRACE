import math
from pathlib import Path

import torch
import torch.nn as nn


CHANNEL_DICT = {
    k: v for v, k in enumerate(
        ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    )
}


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        low = norm_cdf((a - mean) / std)
        high = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * low - 1, 2 * high - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def apply_mask(mask, x):
    batch, num_patches, num_channels, dim = x.shape
    if len(mask.shape) == 2:
        masked_num_patches, masked_num_channels = mask.shape
        mask_keep = mask.reshape((1, masked_num_patches * masked_num_channels, 1)).repeat((batch, 1, dim))
        masked_x = torch.gather(x.reshape((batch, num_patches * num_channels, dim)), dim=-2, index=mask_keep)
        return masked_x.contiguous().view((batch, masked_num_patches, masked_num_channels, dim))

    masked_num_patches = mask.shape[0]
    mask_keep = mask.reshape((1, masked_num_patches, 1)).repeat((batch, 1, dim))
    return torch.gather(x.reshape((batch, num_patches * num_channels, dim)), dim=-2, index=mask_keep)


def apply_mask_t(mask_t, x):
    batch, _, dim = x.shape
    masked_num_patches = mask_t.shape[0]
    mask_keep = mask_t.reshape((1, masked_num_patches, 1)).repeat((batch, 1, dim))
    return torch.gather(x, dim=1, index=mask_keep)


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        batch, tokens, channels = x.shape
        qkv = (
            self.qkv(x)
            .reshape(batch, tokens, 3, self.num_heads, channels // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().view(batch, tokens, channels)
        out = self.proj(out)
        return self.proj_drop(out)


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=(12, 2250), patch_size=75, patch_stride=None, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        if patch_stride is None:
            self.num_patches = (img_size[0], img_size[1] // patch_size)
        else:
            self.num_patches = (img_size[0], (img_size[1] - patch_size) // patch_stride + 1)

        self.proj = nn.Conv2d(
            1,
            embed_dim,
            kernel_size=(1, patch_size),
            stride=(1, patch_size if patch_stride is None else patch_stride),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        return self.proj(x).transpose(1, 3)


class ST2Encoder(nn.Module):
    def __init__(
        self,
        img_size=(12, 2250),
        patch_size=75,
        patch_stride=None,
        embed_dim=256,
        embed_num=1,
        depth=10,
        num_heads=4,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        patch_module=PatchEmbed,
        init_std=0.02,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.embed_num = embed_num
        self.output_dim = embed_num * embed_dim
        self.patch_embed = patch_module(
            img_size=img_size,
            patch_size=patch_size,
            patch_stride=patch_stride,
            embed_dim=embed_dim,
        )
        self.num_patches = self.patch_embed.num_patches
        self.chan_embed = nn.Embedding(len(CHANNEL_DICT), embed_dim)
        drop_path_rates = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path_rates[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.summary_token = nn.Parameter(torch.zeros(1, embed_num, embed_dim))
        self.init_std = init_std

        trunc_normal_(self.summary_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def prepare_chan_ids(self, channels):
        chan_ids = []
        for channel in channels:
            if channel not in CHANNEL_DICT:
                raise KeyError(f'Unknown channel name: {channel}')
            chan_ids.append(CHANNEL_DICT[channel])
        return torch.tensor(chan_ids).unsqueeze_(0).long()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=self.init_std)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv2d):
            trunc_normal_(module.weight, std=self.init_std)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, chan_ids=None, mask_x=None, mask_t=None):
        x = self.patch_embed(x)
        batch, num_steps, num_channels, _ = x.shape
        if chan_ids is None:
            chan_ids = torch.arange(0, num_channels)
        chan_ids = chan_ids.to(x)
        x = x + self.chan_embed(chan_ids.long()).unsqueeze(0)

        if mask_x is not None:
            mask_x = mask_x.to(x.device)
            x = apply_mask(mask_x, x)
            batch, num_steps, num_channels, _ = x.shape

        x = x.flatten(0, 1)
        summary_token = self.summary_token.repeat((x.shape[0], 1, 1))
        x = torch.cat([x, summary_token], dim=1)

        for block in self.blocks:
            x = block(x)

        x = x[:, -summary_token.shape[1]:, :]
        x = self.norm(x)
        x = x.flatten(-2)
        x = x.reshape((batch, num_steps, -1))

        if mask_t is not None:
            mask_t = mask_t.to(x.device)
            x = apply_mask_t(mask_t, x)

        return x.reshape((batch, num_steps, self.embed_num, -1))


def build_st2_v7_encoder(seq_len=2250, patch_size=75, num_leads=12, embed_dim=256, embed_num=1, depth=10, num_heads=4):
    return ST2Encoder(
        img_size=(num_leads, seq_len),
        patch_size=patch_size,
        embed_dim=embed_dim,
        embed_num=embed_num,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
    )


def _extract_model_state(checkpoint):
    model = checkpoint.get('model', checkpoint)
    extracted = {}
    for key, value in model.items():
        if key.startswith('encoder.'):
            extracted[key[len('encoder.'):]] = value
        else:
            extracted[key] = value
    return extracted


def load_st2_encoder_checkpoint(model, checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_state = _extract_model_state(checkpoint)
    state_dict = model.state_dict()
    filtered_state = {}
    skipped = []
    for key, value in model_state.items():
        if key not in state_dict or state_dict[key].shape != value.shape:
            skipped.append(key)
            continue
        filtered_state[key] = value

    msg = model.load_state_dict(filtered_state, strict=False)
    if skipped:
        print(f'Skipped {len(skipped)} incompatible ST2.0 checkpoint keys from {checkpoint_path}.')
    print(f'Loaded ST2.0 encoder checkpoint from: {checkpoint_path}')
    print(msg)
    return msg
