import main
import pandas as pd

# Test the updated print_all_results function
if __name__ == "__main__":
    # Use a small subset of players for testing
    test_df = main.df_minimal_example.iloc[:8].copy()
    list_of_players = [main.Player(test_df.iloc[i]) for i in range(8)]

    # Create a session with both balanced and level preferences
    session_of_rounds = main.SessionOfRounds(
        list_of_players,
        amount_of_rounds=2,
        preferences=["balanced", "level"],
        level_gap_tol=1.5,
        num_iter=50,
        seed=42,
    )

    # Print results with levels to see the happiness gained
    print("Testing updated print_all_results with happiness gained:")
    print("=" * 60)
    session_of_rounds.print_all_results(print_levels=True)
