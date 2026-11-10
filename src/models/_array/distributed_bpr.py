"""Opt-in DDP BPR training. Launch with torchrun, never with the ordinary CLI.

--batch-size is the GLOBAL batch, split across workers. CPU/Gloo is supported
for integration tests; Kaggle uses one NCCL process per CUDA device.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .models import BPRMatrixFactorization, positive_int, validate_pairs
from .torch_training import NegativeSampler, pairwise_loss


class BPRLossModule(torch.nn.Module):
    def __init__(self, model, device):
        super().__init__()
        self.P = torch.nn.Parameter(torch.tensor(model.P, dtype=torch.float32, device=device))
        self.Q = torch.nn.Parameter(torch.tensor(model.Q, dtype=torch.float32, device=device))
        self.reg = model.reg

    def forward(self, users, positive, negative):
        return pairwise_loss(self.P[users], self.Q[positive], self.Q[negative], self.reg)


def fit_distributed(model, interactions, n_users, n_items, *, observed=None,
                    device="cpu", batch_size=1024, validation_score=None, verbose=True):
    """Return the fitted NumPy model on rank zero ONLY; peers return None.

    All ranks must call with identical data, config and callback presence.
    Replicated seeded CPU sampling preserves the single-GPU triple stream.
    GPU forwards/backwards process disjoint slices, without padding examples.
    A zero-weight dummy forward handles an empty final slice safely.
    """
    if not dist.is_initialized():
        raise RuntimeError("Initialize a process group before distributed training")
    if type(model) is not BPRMatrixFactorization:
        raise TypeError("This optional trainer supports plain BPR only")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device(device)
    batch_size = positive_int(batch_size, "batch_size")
    model.n_users = positive_int(n_users, "n_users")
    model.n_items = positive_int(n_items, "n_items")
    model._is_fitted = False
    pairs = validate_pairs(interactions, n_users, n_items)
    observed = pairs if observed is None else validate_pairs(observed, n_users, n_items)
    sampler = NegativeSampler(np.unique(np.concatenate((observed, pairs)), axis=0), n_users, n_items)
    eligible = np.asarray([int(u) not in sampler.full_users for u in pairs[:, 0]], dtype=bool)
    model.skipped_pairs = int((~eligible).sum())
    pairs = pairs[eligible]
    if not len(pairs):
        raise ValueError("No valid BPR triples: all users have seen every catalog item")
    rng = np.random.default_rng(model.seed)
    torch.manual_seed(model.seed)
    model._initialize_parameters(rng)
    module = BPRLossModule(model, device)
    # NumPy initialization is float64; release those host copies after transfer.
    model.P = model.Q = None
    network = DDP(module, device_ids=[device.index] if device.type == "cuda" else None,
                  broadcast_buffers=False, gradient_as_bucket_view=True)
    optimizer = torch.optim.Adam(network.parameters(), lr=model.learning_rate, weight_decay=0.0)
    model.training_info = {
        "backend": "pytorch-ddp", "optimizer": "Adam", "world_size": world,
        "batch_size": batch_size, "global_batch_size": batch_size,
        "max_batch_size_per_rank": math.ceil(batch_size / world),
        "actual_device": str(device), "dtype": "float32", "torch_version": torch.__version__,
        "seed": model.seed, "validation_rank": 0,
        "objective": "mean BPR softplus plus per-triple L2",
        "negative_policy": "uniform unseen TRAIN catalog item; all TRAIN observations excluded",
        "reproducibility": "Same seeded triple stream; distributed floating point reductions may vary.",
    }
    model.history, model.best_epoch = [], 0
    best_value, stale, best_weights = -math.inf, 0, None

    def export():
        model.P = module.P.detach().cpu().numpy().copy()
        model.Q = module.Q.detach().cpu().numpy().copy()
        model._invalidate_scores()
        model._is_fitted = True

    if verbose:
        print(f"DDP worker {rank}/{world} ready on {device}; global batch={batch_size}", flush=True)
    for epoch in range(1, model.epochs + 1):
        started = time.monotonic()
        order = rng.permutation(len(pairs))
        total_loss = torch.zeros((), dtype=torch.float64, device=device)
        local_count = 0
        for start in range(0, len(order), batch_size):
            batch = pairs[order[start:start + batch_size]]
            negative = sampler.sample(batch[:, 0], rng)
            indices = np.arange(rank, len(batch), world)
            count = len(indices)
            local_count += count
            if count == 0:
                indices = np.array([0])  # connected zero gradient, same collective count
            u = torch.as_tensor(batch[indices, 0], dtype=torch.long, device=device)
            i = torch.as_tensor(batch[indices, 1], dtype=torch.long, device=device)
            j = torch.as_tensor(negative[indices], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = network(u, i, j)
            # DDP averages gradients across ranks; correct unequal final slices.
            (loss * (world * count / len(batch))).backward()
            optimizer.step()
            total_loss += loss.detach().double() * count
        dist.all_reduce(total_loss)
        finite = torch.isfinite(module.P).all() & torch.isfinite(module.Q).all() & torch.isfinite(total_loss)
        valid = finite.to(dtype=torch.int32)
        dist.all_reduce(valid, op=dist.ReduceOp.MIN)
        if not valid.item():
            raise FloatingPointError("Training diverged; lower learning_rate")
        training_seconds = time.monotonic() - started
        value = torch.zeros((), device=device, dtype=torch.float64)
        validation_started = time.monotonic()
        if rank == 0:
            export()
            if verbose:
                print(f"Epoch {epoch}: training updates complete; CPU validation starting", flush=True)
            if validation_score is not None:
                value.fill_(float(validation_score(model)))
        dist.broadcast(value, src=0)
        if not math.isfinite(value.item()):
            raise ValueError("Validation metric is not finite")
        row = {"epoch": epoch, "sampled_loss": total_loss.item() / len(pairs),
               "training_seconds": training_seconds,
               "validation_seconds": time.monotonic() - validation_started,
               "global_examples": len(pairs), "rank_examples": local_count}
        if validation_score is not None:
            row["validation_ndcg"] = value.item()
            if value.item() > best_value + 1e-12:
                best_value, stale, model.best_epoch = value.item(), 0, epoch
                if rank == 0:
                    best_weights = {name: getattr(model, name).copy() for name in model.parameter_names}
            else:
                stale += 1
        else:
            model.best_epoch = epoch
        model.history.append(row)
        if rank == 0 and verbose:
            print(row, flush=True)
        if validation_score is not None and stale >= model.patience:
            break
    if rank != 0:
        return None
    if best_weights is not None:
        for name, weights in best_weights.items():
            setattr(model, name, weights)
    model._invalidate_scores()
    model._is_fitted = True
    return model


def main(argv=None):
    from .cli import parser
    from .data import PreparedData, write_json
    from .evaluation import evaluate
    from .models import PopularityRecommender
    from .service import save_bundle

    p = parser()
    args = p.parse_args(["train", *(sys.argv[1:] if argv is None else argv)])
    if args.model != "bpr" or args.features:
        p.error("Distributed entry point supports BPR only; use the ordinary CLI for TextBPR")
    if "LOCAL_RANK" not in os.environ:
        p.error("Launch through python -m torch.distributed.run --standalone --nproc_per_node=2")
    local_rank = int(os.environ["LOCAL_RANK"])
    if args.device == "cpu":
        device, backend = torch.device("cpu"), "gloo"
    else:
        if args.device != "cuda":
            p.error("Use --device cuda (each worker selects its own GPU), or --device cpu for tests")
        if not torch.cuda.is_available() or local_rank >= torch.cuda.device_count():
            p.error("Requested GPU is unavailable; enable GPU T4 x2")
        torch.cuda.set_device(local_rank)
        device, backend = torch.device("cuda", local_rank), "nccl"
    # Validation and final output serialization can be CPU-heavy on large runs.
    dist.init_process_group(backend, timeout=timedelta(hours=3))
    try:
        rank = dist.get_rank()
        if Path(args.out).exists() and any(Path(args.out).iterdir()):
            raise ValueError("--out must be a new/empty folder; existing results are never overwritten")
        data = PreparedData.load(args.data)
        if not len(data.arrays["validation"]):
            raise ValueError("No eligible validation positives; inspect prepared/report.json")
        model = BPRMatrixFactorization(**{key: getattr(args, key) for key in
            ("factors", "learning_rate", "reg", "epochs", "seed", "patience")})
        def validate(candidate):
            return evaluate(candidate, data.arrays["validation"], data.arrays["train_seen"],
                            k=args.k, max_users=args.eval_max_users, seed=args.seed)[f"ndcg@{args.k}"]
        fit_distributed(model, data.arrays["train"], data.n_users, data.n_items,
                        observed=data.arrays["train_seen"], device=device,
                        batch_size=args.batch_size, validation_score=validate)
        if rank == 0:
            print("Training complete. Computing final BPR/popularity validation and saving bundles...", flush=True)
            popularity = PopularityRecommender().fit(data.arrays["train"], data.n_users, data.n_items)
            model.feature_info = None
            model.training_info.update(validation_k=args.k, validation_max_users=args.eval_max_users,
                                       validation_seed=args.seed)
            result = {"model": "bpr", "best_epoch": model.best_epoch,
                "validation": evaluate(model, data.arrays["validation"], data.arrays["train_seen"],
                                       k=args.k, max_users=args.eval_max_users, seed=args.seed),
                "popularity_validation": evaluate(popularity, data.arrays["validation"], data.arrays["train_seen"],
                                       k=args.k, max_users=args.eval_max_users, seed=args.seed)}
            save_bundle(args.out, model, popularity, data)
            save_bundle(Path(args.out) / "cpu", model, popularity, data, compact=True)
            write_json(Path(args.out) / "validation.json", result)
            print(json.dumps(result, indent=2, allow_nan=False), flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    main()
