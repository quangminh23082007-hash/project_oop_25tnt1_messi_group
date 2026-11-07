"""Frozen multilingual review encoding and verified, train-only feature caches.

The encoder runs during feature preparation. Loading cached features and serving
recommendations do not import PyTorch, Transformers, or Sentence Transformers.
"""
from __future__ import annotations

import json
from numbers import Integral
from pathlib import Path
import time

import numpy as np

from .data import PreparedData, file_sha256, write_json
from .models import positive_int


QWEN_DIMENSIONS = {
    "Qwen/Qwen3-Embedding-0.6B": 1024,
    "Qwen/Qwen3-Embedding-4B": 2560,
    "Qwen/Qwen3-Embedding-8B": 4096,
}


def _normalize_rows(values):
    """Normalize after casting, so half-precision rounding cannot reach TextBPR."""
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values.astype(np.float64), axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 1e-12)


def _qwen_dimensions(model_name, max_tokens, embedding_dimension):
    if model_name not in QWEN_DIMENSIONS:
        raise ValueError("Supported encoders: " + ", ".join(QWEN_DIMENSIONS))
    positive_int(max_tokens, "max_tokens")
    if max_tokens > 32768:
        raise ValueError("Qwen3 Embedding max_tokens must be <= 32768")
    full_dimension = QWEN_DIMENSIONS[model_name]
    dimension = full_dimension if embedding_dimension is None else positive_int(
        embedding_dimension, "embedding_dimension")
    if not 32 <= dimension <= full_dimension:
        raise ValueError(f"embedding_dimension must be between 32 and {full_dimension}")
    return full_dimension, dimension


def validate_cuda_devices(devices):
    import torch
    if isinstance(devices, str) or len(devices) < 2:
        raise ValueError("--devices requires at least two distinct indexed CUDA devices")
    parsed = [torch.device(device) for device in devices]
    if any(device.type != "cuda" or device.index is None for device in parsed):
        raise ValueError("Use indexed CUDA devices, for example --devices cuda:0 cuda:1")
    names = [str(device) for device in parsed]
    if len(set(names)) != len(names):
        raise ValueError("--devices must name distinct GPUs")
    if not torch.cuda.is_available() or any(device.index >= torch.cuda.device_count() for device in parsed):
        raise RuntimeError("Requested GPUs are unavailable. Select T4 x2 in Kaggle; "
                           "use --device cuda:0 for a single GPU.")
    return names


def _make_multi_encoder(model_name, *, devices, revision, max_tokens,
                        embedding_dimension, local_files_only, batch_size,
                        chunk_size, worker_timeout):
    _qwen_dimensions(model_name, max_tokens, embedding_dimension)
    devices = validate_cuda_devices(devices)
    from huggingface_hub import snapshot_download
    from .encoding_pool import ProcessEncoderPool
    # Download/resolve exactly once. No language model is loaded in this parent.
    snapshot = snapshot_download(repo_id=model_name, revision=revision,
        local_files_only=local_files_only,
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"])
    config = {"model_name": model_name, "revision": revision, "max_tokens": max_tokens,
              "embedding_dimension": embedding_dimension, "local_files_only": True,
              "snapshot_path": str(snapshot)}
    print(f"Starting encoder replicas on {', '.join(devices)}; "
          f"GPU batch={batch_size}, job chunk={chunk_size}", flush=True)
    return ProcessEncoderPool(devices, config, batch_size=batch_size,
                              chunk_size=chunk_size, timeout=worker_timeout)


class QwenSentenceEncoder:
    """Use the published SentenceTransformer modules, including last-token pooling.

    Reviews are encoded as documents without a query prompt. This implementation
    deliberately supports the Qwen3 Embedding family; other model families can
    require different prompts, pooling, or maximum lengths.
    """

    def __init__(self, model_name="Qwen/Qwen3-Embedding-4B", *, revision=None,
                 device="cuda:0", max_tokens=256, embedding_dimension=None,
                 local_files_only=False, snapshot_path=None):
        full_dimension, dimension = _qwen_dimensions(model_name, max_tokens, embedding_dimension)
        try:
            import torch
            import transformers
            import sentence_transformers
            from huggingface_hub import snapshot_download
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError("Install requirements-kaggle.txt before encoding reviews") from error

        selected_device = torch.device(device)
        if selected_device.type not in ("cpu", "cuda"):
            raise ValueError("Use device='cpu' or a CUDA device such as 'cuda:0'")
        if selected_device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable. Enable a GPU in Kaggle, or explicitly use cpu")
            if selected_device.index is not None and selected_device.index >= torch.cuda.device_count():
                raise ValueError("Requested CUDA device does not exist")
            torch.cuda.set_device(selected_device.index if selected_device.index is not None
                                  else torch.cuda.current_device())
        dtype = torch.float16 if selected_device.type == "cuda" else torch.float32

        # Resolve a branch/tag to an immutable HF snapshot before loading. Keep the
        # cache outside the exported recommender; model weights are several GB.
        snapshot = Path(snapshot_path) if snapshot_path is not None else Path(snapshot_download(
            repo_id=model_name, revision=revision, local_files_only=local_files_only,
            allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"],
        ))
        modules_file = snapshot / "modules.json"
        pooling_file = snapshot / "1_Pooling" / "config.json"
        if not modules_file.is_file() or not pooling_file.is_file():
            raise ValueError("Qwen snapshot is missing its published SentenceTransformer modules")
        pooling = json.loads(pooling_file.read_text(encoding="utf-8"))
        if not pooling.get("pooling_mode_lasttoken"):
            raise ValueError("Expected Qwen last-token pooling; check the requested model revision")
        self.model = SentenceTransformer(
            str(snapshot), device=str(selected_device), local_files_only=True,
            trust_remote_code=False,
            model_kwargs={"torch_dtype": dtype, "attn_implementation": "sdpa"},
        )
        self.model.max_seq_length = max_tokens
        self.model.tokenizer.padding_side = "left"
        self.model.eval().requires_grad_(False)
        dimension_getter = getattr(self.model, "get_embedding_dimension", None)
        if dimension_getter is None:  # Sentence Transformers 5.x
            dimension_getter = self.model.get_sentence_embedding_dimension
        native_dimension = int(dimension_getter())
        if native_dimension != full_dimension:
            raise ValueError("Unexpected embedding dimension in the Qwen snapshot")
        self.dimension, self.torch = dimension, torch
        self.info = {
            "model": model_name,
            "requested_revision": revision,
            "resolved_revision": snapshot.name,
            "dimension": dimension,
            "native_dimension": native_dimension,
            "max_tokens": max_tokens,
            "pooling": "published SentenceTransformer modules; last-token pooling",
            "modules_sha256": file_sha256(modules_file),
            "pooling_config_sha256": file_sha256(pooling_file),
            "prompt": "",
            "input_role": "review document",
            "normalization": "truncate dimensions, cast float32, L2-normalize",
            "padding_side": "left",
            "attention_implementation": "sdpa",
            "frozen": True,
            "device": str(selected_device),
            "dtype": str(dtype),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "sentence_transformers_version": sentence_transformers.__version__,
        }

    def encode(self, texts):
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        with self.torch.inference_mode():
            # Explicit prompt="" also disables a model's configured default query
            # prompt. Do not prepend E5-style "query:" or "passage:" here.
            values = self.model.encode(
                list(texts), batch_size=len(texts), prompt="", show_progress_bar=False,
                convert_to_numpy=True, normalize_embeddings=False,
            )
        return _normalize_rows(np.asarray(values)[:, :self.dimension])


def _text_source(data_dir, data):
    source = Path(data_dir) / "train_reviews.jsonl"
    text_info = data.report.get("text", {})
    if not source.is_file() or not text_info.get("sha256"):
        raise ValueError("Training review text is missing; run prepare with real reviews first")
    if file_sha256(source) != text_info["sha256"]:
        raise ValueError("Training review text changed; run prepare again")
    return source, text_info["sha256"]


def _review_records(source, data):
    # Sorted integer pair codes use much less RAM than a Python set of tuples.
    observed = data.arrays["train_seen"]
    if data.n_users * data.n_items > np.iinfo(np.int64).max:
        raise ValueError("ID mapping is too large to validate review pairs")
    observed_codes = np.unique(observed[:, 0] * data.n_items + observed[:, 1])
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
                u, i, text = row["user_idx"], row["item_idx"], row["text"]
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(f"Invalid training review on line {line_number}") from error
            if (isinstance(u, bool) or not isinstance(u, Integral) or
                    isinstance(i, bool) or not isinstance(i, Integral) or
                    not 0 <= u < data.n_users or not 0 <= i < data.n_items):
                raise ValueError("Training text has an invalid user/item index")
            code = u * data.n_items + i
            position = int(np.searchsorted(observed_codes, code))
            if position == len(observed_codes) or observed_codes[position] != code:
                raise ValueError("Text contains a pair outside training observed interactions")
            if row.get("game_id", data.item_ids[i]) != data.item_ids[i]:
                raise ValueError("Training review game ID does not match its item index")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Training text must contain nonempty cleaned review strings")
            yield i, text


def build_text_features(data_dir, out, *, model_name="Qwen/Qwen3-Embedding-4B",
                        revision=None, device="cuda:0", batch_size=2, max_tokens=256,
                        max_reviews_per_item=4, embedding_dimension=None,
                        local_files_only=False, encoder=None, devices=None,
                        chunk_size=64, worker_timeout=600):
    """Stream the first capped reviews per item in the prepared file's fixed order.

    Every selected review is a training observation, including negative reviews.
    Review embeddings are normalized, averaged per game, then normalized again.
    Games without training text receive zero features. `encoder` is an injectable
    test/custom adapter with dimension, info, and encode(texts) attributes.
    """
    started = time.perf_counter()
    positive_int(batch_size, "batch_size")
    positive_int(max_reviews_per_item, "max_reviews_per_item")
    positive_int(max_tokens, "max_tokens")
    positive_int(chunk_size, "chunk_size")
    if devices is not None and encoder is not None:
        raise ValueError("Specify either devices or an injected encoder, not both")
    destination = Path(out)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("Feature output must be a new or empty folder")
    data = PreparedData.load(data_dir)
    source, text_sha = _text_source(data_dir, data)
    # Fail before downloading/loading billions of weights when there is no text.
    records = _review_records(source, data)
    first = next(records, None)
    if first is None:
        raise ValueError("No training review text. Train ordinary BPR now, or add real reviews first")
    owned = encoder is None
    if encoder is None and devices is not None:
        encoder = _make_multi_encoder(model_name, devices=devices, revision=revision,
            max_tokens=max_tokens, embedding_dimension=embedding_dimension,
            local_files_only=local_files_only, batch_size=batch_size,
            chunk_size=chunk_size, worker_timeout=worker_timeout)
    elif encoder is None:
        encoder = QwenSentenceEncoder(
            model_name, revision=revision, device=device, max_tokens=max_tokens,
            embedding_dimension=embedding_dimension, local_files_only=local_files_only,
        )
    try:
        return _aggregate_features(data, destination, records, first, encoder,
                                   batch_size, max_reviews_per_item, text_sha, started)
    finally:
        if owned and callable(getattr(encoder, "close", None)):
            encoder.close()


def _aggregate_features(data, destination, records, first, encoder,
                        batch_size, max_reviews_per_item, text_sha, started):
    dimension = positive_int(encoder.dimension, "encoder.dimension")
    dispatch_size = positive_int(getattr(encoder, "dispatch_size", batch_size), "dispatch_size")
    sums = np.zeros((data.n_items, dimension), dtype=np.float32)
    counts = np.zeros(data.n_items, dtype=np.int64)
    pending_texts, pending_items = [], []
    encoded = total = capped = 0
    next_progress = 1000

    def flush():
        nonlocal encoded, next_progress
        if not pending_texts:
            return
        values = np.asarray(encoder.encode(pending_texts), dtype=np.float32)
        if values.shape != (len(pending_texts), dimension) or not np.isfinite(values).all():
            raise ValueError("Encoder returned an invalid shape or non-finite vectors")
        if (np.linalg.norm(values.astype(np.float64), axis=1) <= 1e-12).any():
            raise ValueError("Encoder returned a zero vector for nonempty review text")
        np.add.at(sums, pending_items, _normalize_rows(values))
        encoded += len(pending_texts)
        pending_texts.clear()
        pending_items.clear()
        if encoded >= next_progress:
            print(f"Encoded {encoded:,} reviews in {time.perf_counter() - started:.1f}s", flush=True)
            next_progress = encoded + 1000

    from itertools import chain
    for i, text in chain((first,), records):
        total += 1
        if counts[i] >= max_reviews_per_item:
            capped += 1
            continue
        counts[i] += 1
        pending_texts.append(text)
        pending_items.append(i)
        if len(pending_texts) >= dispatch_size:
            flush()
    flush()
    X = _normalize_rows(sums)
    destination.mkdir(parents=True, exist_ok=True)
    features_file = destination / "features.npz"
    np.savez_compressed(features_file, X=X, review_counts=counts)
    info = {
        "format_version": 1,
        "data_fingerprint": data.report["fingerprint"],
        "train_reviews_sha256": text_sha,
        "features_sha256": file_sha256(features_file),
        "encoder": dict(encoder.info),
        "aggregation": "L2-normalized mean of normalized review vectors",
        "selection": "first N per item in prepared train_reviews.jsonl order",
        "max_reviews_per_item": max_reviews_per_item,
        "batch_size": batch_size,
        "dispatch_size": dispatch_size,
        "train_reviews_available": total,
        "reviews_encoded": encoded,
        "reviews_skipped_by_cap": capped,
        "items_with_text": int((counts > 0).sum()),
        "items_with_zero_features": int((np.linalg.norm(X, axis=1) <= 1e-12).sum()),
        "n_items": data.n_items,
        "dimension": dimension,
        "dtype": str(X.dtype),
        "feature_build_seconds": time.perf_counter() - started,
    }
    write_json(destination / "feature_info.json", info)
    return info


def load_text_features(data_dir, features_dir):
    """Reject stale splits, changed review files, corrupted caches, and bad shapes."""
    data = PreparedData.load(data_dir)
    _, text_sha = _text_source(data_dir, data)
    folder = Path(features_dir)
    info = json.loads((folder / "feature_info.json").read_text(encoding="utf-8"))
    if info.get("format_version") != 1:
        raise ValueError("Unsupported feature cache format")
    if info.get("data_fingerprint") != data.report["fingerprint"]:
        raise ValueError("Features belong to a different split or ID mapping; rebuild them")
    if info.get("train_reviews_sha256") != text_sha:
        raise ValueError("Features belong to different training review text; rebuild them")
    feature_file = folder / "features.npz"
    if file_sha256(feature_file) != info.get("features_sha256"):
        raise ValueError("Feature cache checksum mismatch; rebuild it")
    dimension = positive_int(info.get("dimension"), "feature dimension")
    with np.load(feature_file, allow_pickle=False) as archive:
        X, counts = archive["X"].copy(), archive["review_counts"].copy()
    if (info.get("n_items") != data.n_items or X.shape != (data.n_items, dimension)
            or X.dtype != np.float32 or not np.isfinite(X).all()):
        raise ValueError("Invalid cached feature matrix")
    if (counts.shape != (data.n_items,) or counts.dtype.kind not in "iu" or (counts < 0).any()
            or int(counts.sum()) != info.get("reviews_encoded")):
        raise ValueError("Invalid cached review counts")
    norms = np.linalg.norm(X, axis=1)
    if not np.all(np.isclose(norms, 0, atol=1e-6) | np.isclose(norms, 1, atol=1e-5)):
        raise ValueError("Cached features must have unit norm or be zero for missing text")
    if np.any(X[counts == 0] != 0):
        raise ValueError("Games without review text must have zero features")
    return X, info
