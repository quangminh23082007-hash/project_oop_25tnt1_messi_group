"""Local MiniLM encoding, masked mean pooling, and train-only item features."""
from __future__ import annotations
import json
import time
from numbers import Integral
from pathlib import Path
import numpy as np
from .data import PreparedData, file_sha256
from .models import positive_int


class MiniLMSentenceEncoder:
    def __init__(self, model_dir, device="cpu", max_length=256):
        folder = Path(model_dir)
        required = ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors")
        missing = [name for name in required if not (folder / name).is_file()]
        if missing:
            raise ValueError(f"Encoder folder is missing: {', '.join(missing)}")
        positive_int(max_length, "max_length")
        if max_length > 256:
            raise ValueError("MiniLM max_length must be <= 256")
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        expected = {"model_type": "bert", "hidden_size": 384, "num_hidden_layers": 6, "num_attention_heads": 12}
        if any(config.get(k) != v for k, v in expected.items()):
            raise ValueError("Expected the uploaded all-MiniLM-L6-v2 encoder")
        try:
            import torch
            import transformers
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise RuntimeError("Install requirements-text.txt to encode new review text") from error
        self.torch, self.max_length, self.device = torch, max_length, torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(folder, use_fast=True, local_files_only=True)
        self.model, loading = AutoModel.from_pretrained(folder, local_files_only=True,
            use_safetensors=True, output_loading_info=True)
        missing_weights = [k for k in loading.get("missing_keys", []) if not k.startswith("pooler.")]
        if missing_weights or loading.get("mismatched_keys") or loading.get("error_msgs"):
            raise ValueError(f"Incomplete token encoder weights: {missing_weights or loading}")
        self.model.to(self.device).eval().requires_grad_(False)
        self.dimension = int(self.model.config.hidden_size)
        self.info = {"model": "sentence-transformers/all-MiniLM-L6-v2", "dimension": self.dimension,
            "max_tokens": max_length, "pooling": "attention-masked mean, then L2 normalization",
            "frozen": True, "local_files_only": True, "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            **{name + "_sha256": file_sha256(folder / name) for name in required}}

    def encode(self, texts):
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        inputs = self.tokenizer(list(texts), padding=True, truncation=True,
                                max_length=self.max_length, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.inference_mode():
            tokens = self.model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).to(tokens.dtype)
            pooled = (tokens * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.cpu().numpy().astype(np.float32, copy=False)


def build_item_features(data_dir, out, encoder, batch_size=32, max_reviews_per_item=8):
    """Stream earliest N training reviews per item; N=0 means all."""
    started = time.perf_counter()
    positive_int(batch_size, "batch_size")
    if isinstance(max_reviews_per_item, bool) or not isinstance(max_reviews_per_item, Integral) or max_reviews_per_item < 0:
        raise ValueError("max_reviews_per_item must be >= 0")
    data = PreparedData.load(data_dir)
    text_info = data.report.get("text")
    if not text_info:
        raise ValueError("Run the updated prepare command to export training review text")
    source = Path(data_dir) / "train_reviews.jsonl"
    if not source.is_file():
        raise ValueError("Training text is missing. Use the bundled item_features.npz, or rerun prepare from the original reviews")
    if file_sha256(source) != text_info["sha256"]:
        raise ValueError("Training review text changed; run prepare again")
    out = Path(out)
    if out.suffix.lower() != ".npz" or out.exists():
        raise ValueError("Choose a new .npz feature output")
    dimension = positive_int(encoder.dimension, "encoder.dimension")
    sums = np.zeros((data.n_items, dimension), dtype=np.float64)
    counts = np.zeros(data.n_items, dtype=np.int64)
    observed = set(map(tuple, data.arrays["train_seen"].tolist()))
    cutoff = data.report["config"]["train_end_ms"]
    pending_texts, pending_items = [], []
    encoded = total = capped = 0
    next_progress, last_timestamp = 1000, -1

    def flush():
        nonlocal encoded, next_progress
        if not pending_texts:
            return
        values = np.asarray(encoder.encode(pending_texts), dtype=np.float64)
        if values.shape != (len(pending_texts), dimension) or not np.isfinite(values).all():
            raise ValueError("Invalid encoder output")
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        if (norms <= 1e-12).any():
            raise ValueError("Encoder returned zero vectors for nonempty text")
        np.add.at(sums, pending_items, values / norms)
        encoded += len(pending_texts)
        pending_texts.clear()
        pending_items.clear()
        if encoded >= next_progress:
            print(f"Encoded {encoded:,} reviews in {time.perf_counter()-started:.1f}s", flush=True)
            next_progress = encoded + 1000

    with source.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            u, i, timestamp = row["user_idx"], row["item_idx"], row["timestamp"]
            if (u, i) not in observed or row["parent_asin"] != data.item_ids[i]:
                raise ValueError("Text does not match the training mapping/interactions")
            if not last_timestamp <= timestamp < cutoff:
                raise ValueError("Text must be chronological and strictly before the training cutoff")
            last_timestamp = timestamp
            if not isinstance(row["text"], str) or not row["text"].strip():
                raise ValueError("Expected nonempty cleaned text")
            total += 1
            if max_reviews_per_item and counts[i] >= max_reviews_per_item:
                capped += 1
                continue
            counts[i] += 1
            pending_texts.append(row["text"])
            pending_items.append(i)
            if len(pending_texts) >= batch_size:
                flush()
    flush()
    if not encoded:
        raise ValueError("No training text; use ordinary BPR or supply reviews with text")
    norms = np.linalg.norm(sums, axis=1, keepdims=True)
    X = np.divide(sums, norms, out=np.zeros_like(sums), where=norms > 1e-12).astype(np.float32)
    manifest = {"format_version": 1, "data_fingerprint": data.report["fingerprint"],
        "train_reviews_sha256": text_info["sha256"], "train_end_ms": cutoff,
        "encoder": encoder.info, "aggregation": "normalized mean; earliest N training reviews per item",
        "max_reviews_per_item": int(max_reviews_per_item), "batch_size": batch_size,
        "train_reviews_available": total, "reviews_encoded": encoded, "reviews_skipped_by_cap": capped,
        "items_with_text": int((counts > 0).sum()), "items_with_zero_features": int((norms[:,0] <= 1e-12).sum()),
        "n_items": data.n_items, "dimension": dimension, "feature_build_seconds": time.perf_counter()-started}
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, review_counts=counts, manifest=np.array(json.dumps(manifest, allow_nan=False)))
    return manifest


def load_item_features(path, data):
    with np.load(path, allow_pickle=False) as file:
        X, counts = file["X"].copy(), file["review_counts"].copy()
        info = json.loads(str(file["manifest"].item()))
    if info.get("format_version") != 1:
        raise ValueError("Unsupported feature format")
    if info["data_fingerprint"] != data.report["fingerprint"]:
        raise ValueError("Features belong to a different split or item mapping")
    if info["train_reviews_sha256"] != data.report.get("text", {}).get("sha256"):
        raise ValueError("Features belong to different training review text")
    if X.shape != (data.n_items, info["dimension"]) or not np.isfinite(X).all():
        raise ValueError("Invalid feature matrix")
    if counts.shape != (data.n_items,) or (counts < 0).any():
        raise ValueError("Invalid feature counts")
    norms = np.linalg.norm(X, axis=1)
    if not np.all(np.isclose(norms, 0, atol=1e-6) | np.isclose(norms, 1, atol=1e-5)):
        raise ValueError("Features must be normalized or zero")
    return X, info
