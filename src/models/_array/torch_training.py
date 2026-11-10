"""Batched BPR optimization on CUDA, with ordinary NumPy models for serving.

The pairwise objective matches the reference model. Batched Adam follows a
different optimization trajectory from its single-triple NumPy SGD loop.
PyTorch is imported only when training or explicitly calling pairwise_loss.
"""
from __future__ import annotations

import math

import numpy as np

from .models import (
    BPRMatrixFactorization, TextBPRMatrixFactorization,
    pair_sets, positive_int, validate_pairs,
)


def pairwise_loss(p, qi, qj, reg, *, W=None, xi=None, xj=None, text_reg=0.0):
    """Mean sampled-triple BPR loss, including the reference L2 penalties.

    Inputs p/qi/qj have shape (batch, factors). X is frozen in fit_torch.
    Regularization applies to sampled rows, including repeated row occurrences;
    optimizer weight_decay is deliberately zero.
    """
    import torch
    difference = qi - qj
    if W is not None:
        if xi is None or xj is None:
            raise ValueError("Text loss requires xi and xj")
        difference = difference + (xi - xj) @ W.T
    delta = (p * difference).sum(dim=-1)
    penalty = 0.5 * reg * (p.square().sum(dim=-1) + qi.square().sum(dim=-1)
                           + qj.square().sum(dim=-1))
    loss = (torch.nn.functional.softplus(-delta) + penalty).mean()
    if W is not None:
        loss = loss + 0.5 * text_reg * W.square().sum()
    return loss


class NegativeSampler:
    """Uniform unseen-item draws; all observed items are blocked, not just likes.

    A sorted integer index handles sparse histories in vectorized batches.
    Dense histories use finite complements, avoiding long rejection loops.
    """

    def __init__(self, observed, n_users, n_items):
        self.n_users = positive_int(n_users, "n_users")
        self.n_items = positive_int(n_items, "n_items")
        if self.n_users * self.n_items > np.iinfo(np.int64).max:
            raise ValueError("User/item index product exceeds int64")
        pairs = validate_pairs(observed, self.n_users, self.n_items)
        self.blocked = pair_sets(pairs)
        self.keys = np.sort(pairs[:, 0] * self.n_items + pairs[:, 1])
        self.full_users = {u for u, seen in self.blocked.items()
                           if len(seen) == self.n_items}
        self.dense = {
            u: np.asarray([i for i in range(self.n_items) if i not in seen], dtype=np.int64)
            for u, seen in self.blocked.items()
            if self.n_items / 2 < len(seen) < self.n_items
        }

    def contains(self, users, items):
        keys = np.asarray(users, dtype=np.int64) * self.n_items + items
        positions = np.searchsorted(self.keys, keys)
        # keys is nonempty because observed passed validate_pairs.
        return ((positions < len(self.keys))
                & (self.keys[np.minimum(positions, len(self.keys) - 1)] == keys))

    def sample(self, users, rng):
        users = np.asarray(users)
        if users.ndim != 1 or not np.issubdtype(users.dtype, np.integer):
            raise ValueError("Sampling users must be a one-dimensional integer array")
        if (users < 0).any() or (users >= self.n_users).any():
            raise ValueError("Sampling user index is out of range")
        if any(int(u) in self.full_users for u in np.unique(users)):
            raise ValueError("Cannot sample negatives for a fully observed user")
        result = rng.integers(self.n_items, size=len(users), dtype=np.int64)
        dense_mask = np.zeros(len(users), dtype=bool)
        for u in np.unique(users):
            if int(u) in self.dense:
                mask = users == u
                result[mask] = rng.choice(self.dense[int(u)], size=int(mask.sum()))
                dense_mask |= mask
        rejected = (~dense_mask) & self.contains(users, result)
        while rejected.any():
            result[rejected] = rng.integers(self.n_items, size=int(rejected.sum()))
            rejected = (~dense_mask) & self.contains(users, result)
        return result


def fit_torch(model, interactions, n_users, n_items, *, observed=None,
              item_features=None, device="cuda", batch_size=1024,
              validation_score=None, verbose=True):
    """Train BPR/TextBPR with batched Adam; return the fitted NumPy model.

    validation_score receives the exported NumPy model after every epoch and
    must return a finite metric to maximize. Best validation weights are restored.
    The encoder is never loaded here. Use a separate feature-building process
    so its GPU memory is released before recommender training.
    """
    import torch

    if not isinstance(model, BPRMatrixFactorization):
        raise TypeError("fit_torch requires a BPRMatrixFactorization or TextBPR model")
    model._is_fitted = False
    model.n_users = positive_int(n_users, "n_users")
    model.n_items = positive_int(n_items, "n_items")
    batch_size = positive_int(batch_size, "batch_size")
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("This trainer supports device='cpu' or 'cuda[:index]'")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Enable a Kaggle GPU or use --device cpu.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError("Requested CUDA device does not exist")

    pairs = validate_pairs(interactions, model.n_users, model.n_items)
    observed = pairs if observed is None else validate_pairs(observed, model.n_users, model.n_items)
    # Always block positives, even if a caller omitted some from observed.
    all_observed = np.unique(np.concatenate((observed, pairs)), axis=0)
    sampler = NegativeSampler(all_observed, model.n_users, model.n_items)
    eligible = np.asarray([int(u) not in sampler.full_users for u in pairs[:, 0]], dtype=bool)
    model.skipped_pairs = int((~eligible).sum())
    pairs = pairs[eligible]
    if len(pairs) == 0:
        raise ValueError("No valid BPR triples: all users have seen every catalog item")

    is_text = isinstance(model, TextBPRMatrixFactorization)
    if is_text:
        X = np.asarray(item_features, dtype=np.float32)
        if X.ndim != 2 or X.shape[0] != model.n_items or X.shape[1] == 0:
            raise ValueError("item_features must have shape (n_items, embedding_dimension)")
        if not np.isfinite(X).all():
            raise ValueError("item_features must be finite")
        norms = np.linalg.norm(X, axis=1)
        if not np.all(np.isclose(norms, 0, atol=1e-6) | np.isclose(norms, 1, atol=1e-5)):
            raise ValueError("Item vectors must have unit norm, or be zero for missing text")
        model.X = X.copy()
    elif item_features is not None:
        raise ValueError("item_features may only be supplied for TextBPR")

    rng = np.random.default_rng(model.seed)
    torch.manual_seed(model.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(model.seed)
    model._initialize_parameters(rng)
    parameters = {
        name: torch.nn.Parameter(torch.as_tensor(getattr(model, name), dtype=torch.float32,
                                                device=device).clone())
        for name in model.parameter_names
    }
    text_tensor = torch.as_tensor(model.X, device=device) if is_text else None
    optimizer = torch.optim.Adam(list(parameters.values()), lr=model.learning_rate, weight_decay=0.0)
    model.training_info = {
        "backend": "pytorch", "optimizer": "Adam", "batch_size": batch_size,
        "requested_device": str(device), "actual_device": str(parameters["P"].device),
        "dtype": "float32", "torch_version": torch.__version__,
        "objective": "mean BPR softplus plus per-triple L2 and shared text projection L2",
        "negative_policy": "uniform unseen TRAIN catalog item; all TRAIN observations excluded",
        "seed": model.seed,
        "reproducibility": "Seeded sampling; CUDA kernels may vary across hardware/software.",
    }
    model.history, model.best_epoch = [], 0
    best_value, stale, best_weights = -math.inf, 0, None

    def export_parameters():
        for name, parameter in parameters.items():
            setattr(model, name, parameter.detach().cpu().numpy().copy())
        model._invalidate_scores()
        model._is_fitted = True

    for epoch in range(1, model.epochs + 1):
        order = rng.permutation(len(pairs))
        total_loss = torch.zeros((), device=device, dtype=torch.float64)
        for start in range(0, len(order), batch_size):
            batch = pairs[order[start:start + batch_size]]
            negative = sampler.sample(batch[:, 0], rng)
            u = torch.as_tensor(batch[:, 0], dtype=torch.long, device=device)
            i = torch.as_tensor(batch[:, 1], dtype=torch.long, device=device)
            j = torch.as_tensor(negative, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            text_kwargs = ({"W": parameters["W"], "xi": text_tensor[i], "xj": text_tensor[j],
                            "text_reg": model.text_reg} if is_text else {})
            loss = pairwise_loss(parameters["P"][u], parameters["Q"][i], parameters["Q"][j],
                                 model.reg, **text_kwargs)
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().to(dtype=torch.float64) * len(batch)
        if not bool(torch.isfinite(total_loss)) or not all(
            bool(torch.isfinite(parameter).all()) for parameter in parameters.values()
        ):
            model._is_fitted = False
            raise FloatingPointError("Training diverged; lower learning_rate")
        export_parameters()
        row = {"epoch": epoch, "sampled_loss": float(total_loss.cpu()) / len(pairs)}
        if validation_score is not None:
            value = float(validation_score(model))
            if not math.isfinite(value):
                model._is_fitted = False
                raise ValueError("Validation metric is not finite")
            row["validation_ndcg"] = value
            if value > best_value + 1e-12:
                best_value, stale = value, 0
                model.best_epoch = epoch
                best_weights = {name: getattr(model, name).copy() for name in model.parameter_names}
            else:
                stale += 1
        else:
            model.best_epoch = epoch
        model.history.append(row)
        if verbose:
            print(row, flush=True)
        if validation_score is not None and stale >= model.patience:
            break
    if best_weights is not None:
        for name, value in best_weights.items():
            setattr(model, name, value)
    model._invalidate_scores()
    model._is_fitted = True
    return model
