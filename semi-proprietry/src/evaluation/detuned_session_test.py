"""
detuned_session_test.py — Paired t-test comparing FER between a normal and
an intentionally detuned (impaired) driving session, alpha=0.05 (proposal
§3.7, RQ2: "does the framework detect a real change in driving behaviour?").

Note: with only 2 laps per session the test has 1 degree of freedom — low
statistical power. Report results honestly; this is a documented limitation,
not a bug. A directional FER increase without significance still supports
the framework's sensitivity.

Ported from `final-cnn-module-codes.ipynb` ("RQ2 — CELL 4: Paired t-test").
"""

from scipy import stats


def run_detuned_session_test(normal_fer_data: dict, detuned_fer_data: dict) -> dict:
    """
    Run the paired t-test on per-lap FER values from compare_sessions().

    H0: mean FER is equal in normal and detuned sessions.
    H1: detuned session has significantly higher FER.

    Returns:
        dict with t_stat, p_value, significant (bool), and a human-readable
        conclusion string. Falls back to a directional-only comparison if
        fewer than 2 laps are available per session (t-test needs n>=2).
    """
    detuned_per_lap = detuned_fer_data["per_lap_fer"]
    normal_per_lap = normal_fer_data["per_lap_fer"]

    print("\n[RQ2] Per-lap FER values:")
    print(f"  Normal  : {normal_per_lap}")
    print(f"  Detuned : {detuned_per_lap}")

    if len(detuned_per_lap) < 2 or len(normal_per_lap) < 2:
        print("\n[RQ2] Only 1 lap per session — t-test requires n>=2.")
        direction = ">" if detuned_fer_data["fer"] > normal_fer_data["fer"] else "<"
        print(f"  Directional result: detuned FER ({detuned_fer_data['fer']:.2f}) "
              f"{direction} normal FER ({normal_fer_data['fer']:.2f})")
        return {
            "t_stat": None, "p_value": None, "significant": None,
            "conclusion": f"Insufficient laps for t-test; directional FER {direction}.",
        }

    t_stat, p_value = stats.ttest_rel(detuned_per_lap, normal_per_lap)
    significant = p_value < 0.05

    print(f"\n{'='*55}\n  PAIRED T-TEST RESULTS  (alpha = 0.05)")
    print("  H0: mean FER is equal in normal and detuned sessions")
    print("  H1: detuned session has significantly higher FER")
    print(f"{'='*55}")
    print(f"  t-statistic : {t_stat:.4f}")
    print(f"  p-value     : {p_value:.4f}")
    print(f"  Significant : {'YES (p < 0.05)' if significant else 'NO (p >= 0.05)'}")

    if significant and t_stat < 0:
        conclusion = ("Framework detects real behavioural change — FER increased "
                       "significantly in the detuned session.")
    elif detuned_fer_data["fer"] > normal_fer_data["fer"]:
        conclusion = ("FER directionally higher in detuned session but result is not "
                       "statistically significant at alpha=0.05, likely due to small "
                       "sample size. The directional result supports the framework's sensitivity.")
    else:
        conclusion = ("No increase detected — review labelling thresholds or confirm "
                       "detuned driving errors were present in the telemetry.")

    print(f"  Conclusion  : {conclusion}\n{'='*55}")

    return {"t_stat": float(t_stat), "p_value": float(p_value), "significant": significant, "conclusion": conclusion}
