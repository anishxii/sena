from scripts.knowledge_policy_comparison import run_comparison


def test_knowledge_policy_comparison_returns_all_baselines() -> None:
    results = run_comparison(turns=5, seed=11)

    assert set(results) == {"personalized", "generic", "fixed_no_change", "random"}
    for rows in results.values():
        assert len(rows) == 3
        for row in rows:
            assert row["total_oracle_mastery_gain"] >= 0.0
            assert set(row["action_counts"]) == {
                "no_change",
                "simplify",
                "deepen",
                "summarize",
                "highlight_key_points",
                "worked_example",
                "analogy",
                "step_by_step",
            }
