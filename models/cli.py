from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

from .data import PreparedData, prepare, write_json
from .evaluation import evaluate
from .models import BPRMatrixFactorization, TextBPRMatrixFactorization, PopularityRecommender
from .service import RecommendationService, load_bundle, save_bundle


def train(data_dir, out, factors=32, learning_rate=0.03, reg=0.01,
          epochs=30, seed=42, patience=5, k=10, eval_users=1000, eval_seed=42,
          algorithm="bpr", features=None, text_reg=0.001):
    data = PreparedData.load(data_dir)
    if not len(data.arrays["validation"]):
        raise ValueError("No warm-start validation positives; inspect prepared/report.json or use more data")
    destination = Path(out)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Model folder is not empty; use a new folder for each experiment")
    pop = PopularityRecommender().fit(data.arrays["train"], data.n_users, data.n_items)
    pop_val = evaluate(pop, data.arrays["validation"], data.arrays["train_seen"], k, eval_users, eval_seed)
    extra = {}
    if algorithm == "text-bpr":
        if not features:
            raise ValueError("text-bpr requires --features; run features first")
        from .text_features import load_item_features
        X, info = load_item_features(features, data)
        model = TextBPRMatrixFactorization(factors, learning_rate, reg, epochs, seed, patience, text_reg)
        model.feature_info = info
        extra["item_features"] = X
    elif algorithm == "bpr":
        if features:
            raise ValueError("Use --algorithm text-bpr when supplying --features")
        model = BPRMatrixFactorization(factors, learning_rate, reg, epochs, seed, patience)
    else:
        raise ValueError("algorithm must be bpr or text-bpr")
    metric_name = "text_bpr" if algorithm == "text-bpr" else "bpr_mf"
    def validation_score(current):
        return evaluate(current, data.arrays["validation"], data.arrays["train_seen"],
                        k, eval_users, eval_seed)[f"ndcg@{k}"]
    start = time.perf_counter()
    model.fit(data.arrays["train"], data.n_users, data.n_items,
              observed=data.arrays["train_seen"], validation_score=validation_score, **extra)
    seconds = time.perf_counter() - start
    metrics = {"popularity": pop_val,
               metric_name: evaluate(model, data.arrays["validation"], data.arrays["train_seen"], k, eval_users, eval_seed),
               "best_epoch": model.best_epoch, "training_seconds": seconds,
               "skipped_positive_pairs_without_negatives": model.skipped_pairs,
               "selection_metric": f"validation ndcg@{k}", "eval_users_limit": eval_users,
               "seed": seed, "eval_seed": eval_seed, "data_fingerprint": data.report["fingerprint"],
               "algorithm": algorithm, "feature_info": getattr(model, "feature_info", None)}
    save_bundle(out, model, pop, data)
    write_json(destination / "validation_metrics.json", metrics)
    write_json(destination / "training_history.json", model.history)
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


def evaluate_bundle(data_dir, model_dir, split="test", k=10, max_users=0, seed=42, out=None):
    data = PreparedData.load(data_dir)
    model, pop, info, _ = load_bundle(model_dir)
    if data.report["fingerprint"] != info["data_fingerprint"]:
        raise ValueError("This checkpoint was trained on a different split/mapping")
    if split not in ("validation", "test"):
        raise ValueError("split must be validation or test")
    seen = data.arrays["train_seen"]
    if split == "test":
        seen = np.concatenate([seen, data.arrays["validation_seen"]])
    result = {"split": split, "data_fingerprint": data.report["fingerprint"],
              "holdout_coverage": data.report["splits"][split],
              "popularity": evaluate(pop, data.arrays[split], seen, k, max_users, seed),
              "text_bpr" if isinstance(model, TextBPRMatrixFactorization) else "bpr_mf": evaluate(model, data.arrays[split], seen, k, max_users, seed)}
    if out:
        write_json(out, result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def features(data, encoder, out, batch_size=32, max_reviews_per_item=8, device="cpu", max_tokens=256):
    from .text_features import MiniLMSentenceEncoder, build_item_features
    result = build_item_features(data, out, MiniLMSentenceEncoder(encoder, device, max_tokens), batch_size, max_reviews_per_item)
    print(json.dumps(result, indent=2), flush=True)
    return result


def demo(out):
    """Synthetic data has planted preference groups, not Amazon benchmark results."""
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Demo output directory must be new or empty")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(8)
    base = 1_600_000_000_000
    reviews = []
    metadata = [{"parent_asin": f"SYNTH_ITEM_{i:03d}",
                 "title": f"Synthetic group {i // 12} item {i:03d}",
                 "categories": ["Synthetic demo"]} for i in range(36)]
    for u in range(60):
        group = u % 3
        items = rng.permutation(np.arange(group * 12, (group + 1) * 12))
        for index, i in enumerate(items[:10]):
            phase = 0 if index < 6 else 1 if index < 8 else 2
            reviews.append({"user_id": f"SYNTH_USER_{u:03d}", "parent_asin": f"SYNTH_ITEM_{i:03d}",
                            "rating": 5.0, "timestamp": base + phase * 100000 + u * 20 + index,
                            "text": "Synthetic fixture; not an actual review"})
        reviews.append({"user_id": f"SYNTH_USER_{u:03d}",
                        "parent_asin": f"SYNTH_ITEM_{((group + 1) % 3) * 12:03d}",
                        "rating": 2.0, "timestamp": base + 50000 + u})
    rng.shuffle(reviews)  # Source order need not be chronological.
    for name, records in (("reviews.jsonl", reviews), ("metadata.jsonl", metadata)):
        with (out / name).open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
    prepare(out / "reviews.jsonl", out / "prepared", metadata=out / "metadata.jsonl",
            train_end_ms=base + 100000, validation_end_ms=base + 200000)
    train(out / "prepared", out / "model", factors=16, learning_rate=0.05,
          reg=0.005, epochs=50, patience=8, k=5, eval_users=0)
    evaluate_bundle(out / "prepared", out / "model", k=5, out=out / "test_metrics.json")
    print(json.dumps(RecommendationService.load(out / "model").recommend("SYNTH_USER_000", 5), indent=2))


def build_parser():
    parser = argparse.ArgumentParser(description="From-scratch Popularity + BPR-MF for Amazon reviews")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="Clean review JSONL/JSONL.GZ and create fixed global-time splits")
    p.add_argument("--reviews", required=True)
    p.add_argument("--metadata")
    p.add_argument("--allowed-items", help="Optional text file containing one curated parent_asin per line")
    p.add_argument("--out", required=True)
    p.add_argument("--positive-rating", type=float, default=4.0)
    p.add_argument("--min-user", type=int, default=3)
    p.add_argument("--min-item", type=int, default=2)
    p.add_argument("--user-fraction", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-fraction", type=float, default=0.8)
    p.add_argument("--validation-fraction", type=float, default=0.1)
    p.add_argument("--train-end-ms", type=int)
    p.add_argument("--validation-end-ms", type=int)
    p = sub.add_parser("features", help="Create cached training-only item sentence vectors locally")
    p.add_argument("--data", required=True)
    p.add_argument("--encoder", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-reviews-per-item", type=int, default=8, help="Earliest N training reviews; 0 means all")
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-tokens", type=int, default=256)
    p = sub.add_parser("train", help="Train using train and select checkpoint using validation only")
    p.add_argument("--algorithm", choices=["bpr", "text-bpr"], default="bpr")
    p.add_argument("--features")
    p.add_argument("--text-reg", type=float, default=0.001)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--factors", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--reg", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--eval-users", type=int, default=1000, help="Fixed validation-user sample; 0 means all")
    p.add_argument("--eval-seed", type=int, default=42, help="Keep fixed across model seeds/configurations")
    p = sub.add_parser("evaluate", help="Evaluate the selected checkpoint without refitting")
    p.add_argument("--data", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--split", choices=["validation", "test"], default="test")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--max-users", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out")
    p = sub.add_parser("recommend", help="Load a model and recommend using an ORIGINAL user ID")
    p.add_argument("--model", required=True)
    p.add_argument("--user", required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--exclude", nargs="*", default=[], help="Additional known parent_asin IDs, e.g. live history")
    policy = p.add_mutually_exclusive_group()
    policy.add_argument("--include-seen", dest="include_seen", action="store_true", help="Allow reviewed items (default)")
    policy.add_argument("--exclude-seen", dest="include_seen", action="store_false", help="Hide reviewed items")
    p.set_defaults(include_seen=True)
    p = sub.add_parser("demo", help="Generate synthetic data, prepare, train, evaluate and recommend")
    p.add_argument("--out", default="demo_output")
    return parser


def main(argv=None):
    args = vars(build_parser().parse_args(argv))
    command = args.pop("command")
    try:
        if command == "prepare":
            result = prepare(**args)
            print(json.dumps(result.report, indent=2))
        elif command == "train":
            args["data_dir"] = args.pop("data")
            train(**args)
        elif command == "features":
            features(**args)
        elif command == "evaluate":
            args["data_dir"], args["model_dir"] = args.pop("data"), args.pop("model")
            evaluate_bundle(**args)
        elif command == "recommend":
            result = RecommendationService.load(args["model"]).recommend(args["user"], args["k"], args["exclude"], include_seen=args["include_seen"])
            print(json.dumps(result, indent=2, ensure_ascii=False))
        elif command == "demo":
            demo(**args)
    except (ValueError, RuntimeError, OSError, FloatingPointError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0
