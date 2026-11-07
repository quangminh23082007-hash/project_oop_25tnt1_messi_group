"""TextBPR requires train-only review features with matching provenance."""
from src.contracts.dto import ItemFeatures
from src.contracts.exceptions import ConfigurationError, EmptyDatasetError
from ._adapter import ArrayModelAdapter, engine_options
from ._array.text_bpr import TextBPRMatrixFactorization as ArrayTextBPR, text_bpr_loss_and_gradients


class TextBPRMatrixFactorization(ArrayModelAdapter):
    def __init__(self, config=None, *, factors=None, learning_rate=None, reg=None,
                 epochs=None, seed=None, patience=None, text_reg=None, **unsupported):
        super().__init__("TextBPR")
        if unsupported:
            raise ConfigurationError(f"Unsupported TextBPR parameters: {', '.join(sorted(unsupported))}")
        self._engine = ArrayTextBPR(**engine_options(
            config, factors=factors, learning_rate=learning_rate, reg=reg,
            epochs=epochs, seed=seed, patience=patience, text_reg=text_reg))

    def _feature_options(self, dataset, item_features):
        if not isinstance(item_features, ItemFeatures):
            raise ConfigurationError("TextBPR requires shared ItemFeatures, including train/text provenance")
        if next(iter(dataset.iter_reviews()), None) is None:
            raise EmptyDatasetError("TextBPR requires nonblank train reviews, not just a feature matrix")
        if (item_features.mapping_fingerprint != dataset.mapping_fingerprint
                or item_features.train_fingerprint != dataset.fingerprint
                or item_features.text_fingerprint != dataset.review_fingerprint):
            raise ConfigurationError("Feature mapping/train/text fingerprints do not match train")
        ordered_ids = tuple(dataset.item_mapper.to_raw(i) for i in range(dataset.num_items))
        if item_features.item_ids != ordered_ids:
            raise ConfigurationError("Feature item_ids are not in the train mapping order")
        # Revalidate the boundary, even if a caller forcibly mutated the frozen DTO.
        checked = ItemFeatures(item_features.values, item_features.item_ids,
                               item_features.mapping_fingerprint, item_features.train_fingerprint,
                               item_features.text_fingerprint, item_features.encoder_name)
        return {"item_features": checked.values}


__all__ = ["TextBPRMatrixFactorization", "text_bpr_loss_and_gradients"]
