"""Compatibility exports for callers of the previous monolithic models module."""
if __package__:
    from .base import BaseRecommender
    from .matrix_factorization import (
        BPRMatrixFactorization,
        MatrixFactorizationRecommender,
        bpr_loss_and_gradients,
    )
    from .popularity import PopularityRecommender
    from .text_bpr import TextBPRMatrixFactorization, text_bpr_loss_and_gradients
    from ._utils import pair_sets, positive_int, validate_pairs
else:
    from base import BaseRecommender
    from matrix_factorization import (
        BPRMatrixFactorization,
        MatrixFactorizationRecommender,
        bpr_loss_and_gradients,
    )
    from popularity import PopularityRecommender
    from text_bpr import TextBPRMatrixFactorization, text_bpr_loss_and_gradients
    from _utils import pair_sets, positive_int, validate_pairs

__all__ = [
    "BaseRecommender",
    "BPRMatrixFactorization",
    "MatrixFactorizationRecommender",
    "PopularityRecommender",
    "TextBPRMatrixFactorization",
    "bpr_loss_and_gradients",
    "text_bpr_loss_and_gradients",
    "pair_sets",
    "positive_int",
    "validate_pairs",
]
