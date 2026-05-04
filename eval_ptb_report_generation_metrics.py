
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


SPECIAL_TOKEN_RE = re.compile(r"<\|[^>]+?\|>")
WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate PTB-XL report-generation NLG metrics.")
    parser.add_argument("--input-json", type=str, default="data/generated_ptbxl_st2_reports.json")
    parser.add_argument("--output-json", type=str, default="data/generated_ptbxl_st2_nlg_metrics.json")
    parser.add_argument("--pred-key", type=str, default="clean_report")
    parser.add_argument("--ref-key", type=str, default="reference_report")
    parser.add_argument("--section", choices=["report", "full"], default="report")
    parser.add_argument("--keep-special-tokens", action="store_true")
    parser.add_argument("--bertscore", action="store_true")
    parser.add_argument("--bertscore-model", type=str, default="bert-base-uncased")
    parser.add_argument("--bertscore-batch-size", type=int, default=32)
    parser.add_argument("--bertscore-device", type=str, default="cuda")
    parser.add_argument("--bertscore-rescale-with-baseline", action="store_true")
    parser.add_argument("--metric-backend", choices=["local", "pycocoevalcap"], default="local")
    parser.add_argument("--write-normalized-jsonl", type=str, default="")
    return parser.parse_args()


def extract_report_text(text: str, section: str) -> str:
    text = str(text or "")
    if section == "report" and "<|report_start|>" in text:
        text = text.split("<|report_start|>", 1)[1]
    for marker in ("<|end|>", "<|assistant|>", "<|user|>"):
        if marker in text:
            text = text.split(marker, 1)[0]
    return text


def normalize_text(text: str, section: str = "report", keep_special_tokens: bool = False) -> str:
    text = extract_report_text(text, section=section)
    if not keep_special_tokens:
        text = SPECIAL_TOKEN_RE.sub(" ", text)
    text = text.replace("#", " ").replace("*", " ").replace("-", " ")
    text = re.sub(r"[_/\\|]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(pred_tokens: list[list[str]], ref_tokens: list[list[str]], max_order: int = 4) -> dict[str, float]:
    hyp_len = sum(len(tokens) for tokens in pred_tokens)
    ref_len = sum(len(tokens) for tokens in ref_tokens)
    matches = [0] * max_order
    totals = [0] * max_order

    for pred, ref in zip(pred_tokens, ref_tokens):
        for n in range(1, max_order + 1):
            pred_counts = ngrams(pred, n)
            ref_counts = ngrams(ref, n)
            overlap = pred_counts & ref_counts
            matches[n - 1] += sum(overlap.values())
            totals[n - 1] += max(sum(pred_counts.values()), 0)

    precisions = []
    for match, total in zip(matches, totals):
        if total == 0:
            precisions.append(0.0)
        elif match == 0:
            precisions.append(1.0 / (2.0 * total))
        else:
            precisions.append(match / total)

    if hyp_len == 0:
        bp = 0.0
    elif hyp_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1.0 - ref_len / max(hyp_len, 1))

    bleu1 = bp * precisions[0]
    if min(precisions[:4]) <= 0.0:
        bleu4 = 0.0
    else:
        bleu4 = bp * math.exp(sum(math.log(p) for p in precisions[:4]) / 4.0)
    return {
        "BLEU-1": bleu1 * 100.0,
        "BLEU-4": bleu4 * 100.0,
        "brevity_penalty": bp,
        "hyp_tokens": float(hyp_len),
        "ref_tokens": float(ref_len),
    }


def lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    if len(a) < len(b):
        short, long = a, b
    else:
        short, long = b, a
    prev = [0] * (len(short) + 1)
    for token in long:
        curr = [0]
        for j, other in enumerate(short, start=1):
            if token == other:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[-1]))
        prev = curr
    return prev[-1]


def rouge_l(pred_tokens: list[list[str]], ref_tokens: list[list[str]]) -> float:
    scores = []
    for pred, ref in zip(pred_tokens, ref_tokens):
        if not pred or not ref:
            scores.append(0.0)
            continue
        lcs = lcs_length(pred, ref)
        precision = lcs / len(pred)
        recall = lcs / len(ref)
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(2.0 * precision * recall / (precision + recall))
    return float(np.mean(scores) * 100.0)


def exact_meteor(pred: list[str], ref: list[str]) -> float:
    if not pred or not ref:
        return 0.0
    ref_positions: dict[str, list[int]] = defaultdict(list)
    for idx, token in enumerate(ref):
        ref_positions[token].append(idx)

    used = set()
    alignments = []
    for pred_idx, token in enumerate(pred):
        for ref_idx in ref_positions.get(token, []):
            if ref_idx not in used:
                used.add(ref_idx)
                alignments.append((pred_idx, ref_idx))
                break

    matches = len(alignments)
    if matches == 0:
        return 0.0
    precision = matches / len(pred)
    recall = matches / len(ref)
    fmean = (10.0 * precision * recall) / (recall + 9.0 * precision) if precision and recall else 0.0

    alignments.sort()
    chunks = 1
    for prev, curr in zip(alignments, alignments[1:]):
        if curr[0] != prev[0] + 1 or curr[1] != prev[1] + 1:
            chunks += 1
    penalty = 0.5 * (chunks / matches) ** 3
    return (1.0 - penalty) * fmean


def meteor(pred_tokens: list[list[str]], ref_tokens: list[list[str]]) -> float:
    return float(np.mean([exact_meteor(pred, ref) for pred, ref in zip(pred_tokens, ref_tokens)]) * 100.0)


def document_frequency(ref_tokens: list[list[str]], max_order: int = 4) -> dict[int, Counter[tuple[str, ...]]]:
    dfs: dict[int, Counter[tuple[str, ...]]] = {n: Counter() for n in range(1, max_order + 1)}
    for ref in ref_tokens:
        for n in range(1, max_order + 1):
            dfs[n].update(set(ngrams(ref, n)))
    return dfs


def tfidf_vector(tokens: list[str], n: int, dfs: Counter[tuple[str, ...]], num_docs: int) -> dict[tuple[str, ...], float]:
    counts = ngrams(tokens, n)
    if not counts:
        return {}
    total = sum(counts.values())
    vector = {}
    for gram, count in counts.items():
        tf = count / total
        df = max(float(dfs.get(gram, 0)), 1.0)
        idf = math.log(max(float(num_docs), 1.0) / df)
        vector[gram] = tf * idf
    return vector


def cosine(a: dict[tuple[str, ...], float], b: dict[tuple[str, ...], float]) -> float:
    if not a or not b:
        return 0.0
    common = set(a).intersection(b)
    numerator = sum(a[key] * b[key] for key in common)
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return numerator / (norm_a * norm_b)


def cider_d(pred_tokens: list[list[str]], ref_tokens: list[list[str]], sigma: float = 6.0) -> float:
    num_docs = len(ref_tokens)
    dfs = document_frequency(ref_tokens, max_order=4)
    scores = []
    for pred, ref in zip(pred_tokens, ref_tokens):
        length_penalty = math.exp(-((len(pred) - len(ref)) ** 2) / (2.0 * sigma * sigma))
        order_scores = []
        for n in range(1, 5):
            pred_vec = tfidf_vector(pred, n, dfs[n], num_docs)
            ref_vec = tfidf_vector(ref, n, dfs[n], num_docs)
            order_scores.append(cosine(pred_vec, ref_vec) * length_penalty)
        scores.append(10.0 * float(np.mean(order_scores)))
    return float(np.mean(scores))


def bertscore_f1(preds: list[str], refs: list[str], args: argparse.Namespace) -> dict[str, float | str]:
    try:
        from bert_score import score as bert_score
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"BERTScore-F1": None, "bertscore_error": f"bert_score import failed: {exc}"}

    try:
        _p, _r, f1 = bert_score(
            preds,
            refs,
            lang="en",
            model_type=args.bertscore_model,
            batch_size=args.bertscore_batch_size,
            device=args.bertscore_device,
            verbose=True,
            rescale_with_baseline=args.bertscore_rescale_with_baseline,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"BERTScore-F1": None, "bertscore_error": str(exc)}
    return {"BERTScore-F1": float(f1.mean().item() * 100.0)}


def pycocoevalcap_metrics(preds: list[str], refs: list[str]) -> dict[str, float | str]:
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.rouge.rouge import Rouge
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"pycocoevalcap_error": f"import failed: {exc}"}

    gts = {idx: [ref] for idx, ref in enumerate(refs)}
    res = {idx: [pred] for idx, pred in enumerate(preds)}
    metrics: dict[str, float | str] = {}

    scorer_specs = [
        (lambda: Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]),
        (lambda: Meteor(), "METEOR"),
        (lambda: Rouge(), "ROUGE-L"),
        (lambda: Cider(), "CIDEr-D"),
    ]
    for scorer_factory, names in scorer_specs:
        try:
            scorer = scorer_factory()
        except Exception as exc:  # pragma: no cover - environment dependent
            metrics[f"{names}_error" if isinstance(names, str) else "BLEU_error"] = str(exc)
            continue
        try:
            score, _scores = scorer.compute_score(gts, res)
            if isinstance(names, list):
                for name, value in zip(names, score):
                    metrics[name] = float(value * 100.0)
            elif names == "CIDEr-D":
                metrics[names] = float(score)
            else:
                metrics[names] = float(score * 100.0)
        except Exception as exc:  # pragma: no cover - environment dependent
            metrics[f"{names}_error" if isinstance(names, str) else "BLEU_error"] = str(exc)
        finally:
            if hasattr(scorer, "close"):
                try:
                    scorer.close()
                except Exception:
                    pass
    return metrics


def load_pairs(args: argparse.Namespace) -> tuple[list[str], list[str], list[dict]]:
    with Path(args.input_json).open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise TypeError(f"Expected a JSON list from {args.input_json}, got {type(rows)!r}")

    preds = []
    refs = []
    normalized_rows = []
    for idx, row in enumerate(rows):
        pred = normalize_text(row.get(args.pred_key, ""), section=args.section, keep_special_tokens=args.keep_special_tokens)
        ref = normalize_text(row.get(args.ref_key, ""), section=args.section, keep_special_tokens=args.keep_special_tokens)
        preds.append(pred)
        refs.append(ref)
        normalized_rows.append(
            {
                "index": idx,
                "ecg_id": row.get("ecg_id"),
                "prediction": pred,
                "reference": ref,
            }
        )
    return preds, refs, normalized_rows


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    preds, refs, normalized_rows = load_pairs(args)
    pred_tokens = [tokenize(text) for text in preds]
    ref_tokens = [tokenize(text) for text in refs]

    metrics = {
        "num_samples": len(preds),
        "input_json": args.input_json,
        "pred_key": args.pred_key,
        "ref_key": args.ref_key,
        "section": args.section,
        "keep_special_tokens": args.keep_special_tokens,
        "avg_pred_tokens": float(np.mean([len(tokens) for tokens in pred_tokens])) if pred_tokens else 0.0,
        "avg_ref_tokens": float(np.mean([len(tokens) for tokens in ref_tokens])) if ref_tokens else 0.0,
    }
    metrics["metric_backend"] = args.metric_backend
    if args.metric_backend == "pycocoevalcap":
        coco_metrics = pycocoevalcap_metrics(preds, refs)
        metrics.update(coco_metrics)
        if any(key.endswith("_error") for key in coco_metrics):
            metrics["local_fallback_used"] = True
            local_metrics = corpus_bleu(pred_tokens, ref_tokens)
            local_metrics["METEOR"] = meteor(pred_tokens, ref_tokens)
            local_metrics["ROUGE-L"] = rouge_l(pred_tokens, ref_tokens)
            local_metrics["CIDEr-D"] = cider_d(pred_tokens, ref_tokens)
            metrics.update({f"local_{key}": value for key, value in local_metrics.items()})
    else:
        metrics.update(corpus_bleu(pred_tokens, ref_tokens))
        metrics["METEOR"] = meteor(pred_tokens, ref_tokens)
        metrics["ROUGE-L"] = rouge_l(pred_tokens, ref_tokens)
        metrics["CIDEr-D"] = cider_d(pred_tokens, ref_tokens)

    if args.bertscore:
        metrics.update(
            {
                "bertscore_model": args.bertscore_model,
                "bertscore_batch_size": args.bertscore_batch_size,
                "bertscore_device": args.bertscore_device,
                "bertscore_rescale_with_baseline": args.bertscore_rescale_with_baseline,
            }
        )
        metrics.update(bertscore_f1(preds, refs, args))

    output_path = Path(args.output_json)
    write_json(output_path, metrics)
    if args.write_normalized_jsonl:
        write_jsonl(Path(args.write_normalized_jsonl), normalized_rows)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
