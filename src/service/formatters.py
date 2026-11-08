"""Presentation formatters for recommendation DTOs."""

from typing import Sequence

from src.contracts.dto import RecommendationResult


def format_recommendations_table(results: Sequence[RecommendationResult]) -> str:
    if not results:
        return "No matching game recommendations found."
    header = (f"{'Rank':<6} | {'Game ID':<10} | {'Title':<24} | {'Genres':<22} | "
              f"{'Price':<8} | {'Developer':<16} | {'Score':>8} | {'Fallback':<8}")
    rows = [header, "-" * len(header)]
    for r in results:
        genre_str = ", ".join(r.genres)
        price_str = f"${r.price:.2f}" if r.price is not None else "N/A"
        rows.append(f"#{r.rank:<5} | {r.game_id:<10} | {r.title[:24]:<24} | "
                    f"{genre_str[:22]:<22} | {price_str:<8} | {r.developer[:16]:<16} | "
                    f"{r.predicted_score:>8.4f} | {str(r.is_fallback):<8}")
    return "\n".join(rows)
