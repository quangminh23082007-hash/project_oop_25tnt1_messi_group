"""Service layer module complying with Contracts v1 and v1.1 extensions."""

from src.service.filters import FilterCriteria, game_passes_filters, parse_filter_criteria
from src.service.formatters import format_recommendations_table
from src.service.mappers import to_recommendation_result
from src.service.pretrained import PretrainedDatasetProxy, PretrainedServiceFacade
from src.service.recommendation_service import RecommendationService

__all__ = [
    "RecommendationService",
    "PretrainedServiceFacade",
    "PretrainedDatasetProxy",
    "FilterCriteria",
    "parse_filter_criteria",
    "game_passes_filters",
    "format_recommendations_table",
    "to_recommendation_result",
]
