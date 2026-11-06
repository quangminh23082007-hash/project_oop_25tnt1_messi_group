"""Command line entry point for the Service layer.

Usage:
    python -m src.service
"""

from src.ui.bootstrap import create_demo_service


def main() -> None:
    print("=" * 60)
    print("STEAM GAME RECOMMENDATION SERVICE - DEMO EXECUTION")
    print("=" * 60)
    service = create_demo_service()
    print(f"Service loaded successfully. Primary model: {service.model_name}")
    print(f"Fallback model: {service.fallback_model.name}")
    print(f"Is ready: {service.is_ready}")
    print("-" * 60)

    user_id = "steam_user_101"
    print(f"Generating recommendations for: {user_id}")
    results = service.recommend_games(user_id, top_k=3)
    print(service.format_recommendations_table(results))

    print("\n" + "=" * 60)
    cold_user = "steam_user_cold"
    print(f"Generating cold-start recommendations for: {cold_user}")
    cold_results = service.recommend_games(cold_user, top_k=3)
    print(service.format_recommendations_table(cold_results))
    print("=" * 60)


if __name__ == "__main__":
    main()
