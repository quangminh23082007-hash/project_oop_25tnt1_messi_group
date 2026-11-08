"""Contracts v1 adapter for BPR trained by NumPy SGD."""
from src.contracts.exceptions import ConfigurationError
from ._adapter import ArrayModelAdapter, engine_options
from ._array.matrix_factorization import BPRMatrixFactorization as ArrayBPR, bpr_loss_and_gradients


class BPRMatrixFactorization(ArrayModelAdapter):
    def __init__(self, config=None, *, factors=None, learning_rate=None, reg=None,
                 epochs=None, seed=None, patience=None, **unsupported):
        super().__init__("BPR")
        if unsupported:
            raise ConfigurationError(f"Unsupported BPR parameters: {', '.join(sorted(unsupported))}")
        options = engine_options(config, factors=factors, learning_rate=learning_rate,
                                 reg=reg, epochs=epochs, seed=seed, patience=patience)
        options.pop("text_reg")  # Shared config field only applies to TextBPR.
        self._engine = ArrayBPR(**options)


MatrixFactorizationRecommender = BPRMatrixFactorization
__all__ = ["BPRMatrixFactorization", "MatrixFactorizationRecommender", "bpr_loss_and_gradients"]
