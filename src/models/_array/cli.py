"""CLI for data preparation, offline encoding, GPU training and CPU serving."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Prepare Steam-style interactions and train-only review text")
    prep.add_argument("--interactions", required=True)
    prep.add_argument("--games")
    prep.add_argument("--reviews")
    prep.add_argument("--out", required=True)
    prep.add_argument("--split", choices=("random", "chronological"), default="random")
    identity = prep.add_mutually_exclusive_group(required=True)
    identity.add_argument("--allow-unverified-users", action="store_true")
    identity.add_argument("--verified-user-ids", action="store_true")
    prep.add_argument("--user-fraction", type=float, default=1.0)
    prep.add_argument("--min-user", type=int, default=3)
    prep.add_argument("--min-item", type=int, default=2)
    prep.add_argument("--train-fraction", type=float, default=.8)
    prep.add_argument("--validation-fraction", type=float, default=.1)
    prep.add_argument("--seed", type=int, default=42)

    features = commands.add_parser("build-text-features", help="Run frozen Qwen on train review documents")
    features.add_argument("--data", required=True)
    features.add_argument("--out", required=True)
    features.add_argument("--model-name", default="Qwen/Qwen3-Embedding-4B")
    features.add_argument("--revision", help="Pin a Hugging Face commit for repeatable encoding")
    feature_devices = features.add_mutually_exclusive_group()
    feature_devices.add_argument("--device", default="cuda:0")
    feature_devices.add_argument("--devices", nargs="+", help="Encode on replicas, e.g. cuda:0 cuda:1")
    features.add_argument("--batch-size", type=int, default=2)
    features.add_argument("--chunk-size", type=int, default=64, help="Texts dispatched per encoder worker")
    features.add_argument("--worker-timeout", type=float, default=600, help="Seconds allowed for worker startup or a dispatched chunk")
    features.add_argument("--max-tokens", type=int, default=256)
    features.add_argument("--max-reviews-per-item", type=int, default=4)
    features.add_argument("--embedding-dimension", type=int)
    features.add_argument("--local-files-only", action="store_true")

    train = commands.add_parser("train", help="Train batched PyTorch BPR or TextBPR")
    train.add_argument("--data", required=True)
    train.add_argument("--out", required=True)
    train.add_argument("--model", choices=("bpr", "text-bpr"), default="bpr")
    train.add_argument("--features")
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--batch-size", type=int, default=1024)
    train.add_argument("--factors", type=int, default=64)
    train.add_argument("--learning-rate", type=float, default=.001)
    train.add_argument("--reg", type=float, default=.01)
    train.add_argument("--text-reg", type=float, default=.001)
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--patience", type=int, default=5)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--k", type=int, default=10)
    train.add_argument("--eval-max-users", type=int, default=1000,
                       help="Fixed validation user sample, 0 means all")

    ev = commands.add_parser("evaluate", help="Evaluate against the identical prepared data split")
    ev.add_argument("--data", required=True)
    ev.add_argument("--model", required=True)
    ev.add_argument("--split", choices=("validation", "test"), default="validation")
    ev.add_argument("--k", type=int, default=10)
    ev.add_argument("--max-users", type=int, default=0)
    ev.add_argument("--seed", type=int, default=42)
    ev.add_argument("--out")

    recommend = commands.add_parser("recommend", help="Serve existing catalog recommendations with NumPy")
    recommend.add_argument("--model", required=True)
    recommend.add_argument("--user", required=True)
    recommend.add_argument("--k", type=int, default=10)
    recommend.add_argument("--extra-seen", nargs="*", default=[], metavar="GAME_ID")
    recommend.add_argument("--include-seen", action="store_true")
    return root


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.command == "prepare":
        from .data import prepare
        kwargs = vars(args).copy()
        kwargs.pop("command")
        result = prepare(**kwargs)
        output = result.report
    elif args.command == "build-text-features":
        from .text_features import build_text_features
        kwargs = vars(args).copy()
        kwargs.pop("command")
        kwargs["data_dir"] = kwargs.pop("data")
        output = build_text_features(**kwargs)
    elif args.command == "train":
        from .data import PreparedData, write_json
        from .evaluation import evaluate
        from .models import BPRMatrixFactorization, TextBPRMatrixFactorization, PopularityRecommender
        from .service import save_bundle
        from .torch_training import fit_torch
        if Path(args.out).exists() and any(Path(args.out).iterdir()):
            p.error("--out must be a new/empty folder")
        if args.model == "text-bpr" and not args.features:
            p.error("text-bpr requires --features; encode real training reviews first")
        if args.model == "bpr" and args.features:
            p.error("--features is only used by --model text-bpr")
        data = PreparedData.load(args.data)
        if not len(data.arrays["validation"]):
            p.error("No eligible validation positives; inspect report, increase sample or reduce min counts")
        params = {key: getattr(args, key) for key in
                  ("factors", "learning_rate", "reg", "epochs", "seed", "patience")}
        X = info = None
        if args.model == "text-bpr":
            from .text_features import load_text_features
            X, info = load_text_features(args.data, args.features)
            model = TextBPRMatrixFactorization(**params, text_reg=args.text_reg)
        else:
            model = BPRMatrixFactorization(**params)
        popularity = PopularityRecommender().fit(data.arrays["train"], data.n_users, data.n_items)
        def validate(candidate):
            return evaluate(candidate, data.arrays["validation"], data.arrays["train_seen"],
                            k=args.k, max_users=args.eval_max_users, seed=args.seed)[f"ndcg@{args.k}"]
        fit_torch(model, data.arrays["train"], data.n_users, data.n_items,
                  observed=data.arrays["train_seen"], item_features=X,
                  device=args.device, batch_size=args.batch_size, validation_score=validate)
        model.feature_info = info
        model.training_info.update(validation_k=args.k, validation_max_users=args.eval_max_users,
                                   validation_seed=args.seed)
        output = {"model": args.model, "best_epoch": model.best_epoch,
                  "validation": evaluate(model, data.arrays["validation"], data.arrays["train_seen"],
                                         k=args.k, max_users=args.eval_max_users, seed=args.seed),
                  "popularity_validation": evaluate(popularity, data.arrays["validation"],
                      data.arrays["train_seen"], k=args.k, max_users=args.eval_max_users, seed=args.seed)}
        save_bundle(args.out, model, popularity, data)
        save_bundle(Path(args.out) / "cpu", model, popularity, data, compact=True)
        write_json(Path(args.out) / "validation.json", output)
    elif args.command == "evaluate":
        from .data import PreparedData, write_json
        from .evaluation import evaluate
        from .service import load_bundle
        data = PreparedData.load(args.data)
        model, popularity, info, _ = load_bundle(args.model)
        if (info["data_fingerprint"] != data.report["fingerprint"] or
                info["user_ids"] != data.user_ids or info["item_ids"] != data.item_ids):
            p.error("Model and prepared data do not match")
        seen = data.arrays["validation_seen" if args.split == "test" else "train_seen"]
        output = {"split": args.split,
                  "model": evaluate(model, data.arrays[args.split], seen, args.k, args.max_users, args.seed),
                  "popularity": evaluate(popularity, data.arrays[args.split], seen, args.k, args.max_users, args.seed)}
        if args.out:
            write_json(args.out, output)
    else:
        from .service import RecommendationService
        output = RecommendationService.load(args.model).recommend(
            args.user, args.k, args.extra_seen, include_seen=args.include_seen)
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))
    return 0
