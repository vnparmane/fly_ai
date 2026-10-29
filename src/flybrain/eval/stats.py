"""Statistical analysis and hypothesis testing for controller evaluation."""

import numpy as np
import pandas as pd
import scipy.stats as stats


def bootstrap_ci(data: np.ndarray, n_boot: int = 10000, ci: float = 0.95, seed: int = 42) -> tuple[float, float, float]:
    """Compute bootstrap mean and confidence interval."""
    rng = np.random.default_rng(seed)
    data = np.asarray(data, dtype=np.float64)
    if len(data) == 0:
        return 0.0, 0.0, 0.0

    boot_means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample = rng.choice(data, size=len(data), replace=True)
        boot_means[i] = np.mean(sample)

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, alpha * 100.0))
    upper = float(np.percentile(boot_means, (1.0 - alpha) * 100.0))
    mean = float(np.mean(data))
    return mean, lower, upper


def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cohen's d effect size between two groups."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 0.0
    s_pooled = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
    if s_pooled == 0:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / s_pooled)


def compute_evaluation_stats(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Compute summary statistics and statistical comparison markdown table."""
    controllers = df["controller"].unique()
    summary_rows = []

    for ctrl in controllers:
        sub = df[df["controller"] == ctrl]
        kills = sub["kills"].values
        steps = sub["steps_survived"].values
        mean_k, low_k, high_k = bootstrap_ci(kills)
        mean_s, low_s, high_s = bootstrap_ci(steps)

        shots = sub["shots_fired"].values
        acc = kills / np.maximum(1, shots)
        mean_acc = float(np.mean(acc))

        summary_rows.append({
            "controller": ctrl,
            "n_episodes": len(sub),
            "kills_mean": mean_k,
            "kills_ci95": f"[{low_k:.2f}, {high_k:.2f}]",
            "steps_mean": mean_s,
            "steps_ci95": f"[{low_s:.1f}, {high_s:.1f}]",
            "accuracy": f"{mean_acc*100:.1f}%",
        })

    summary_df = pd.DataFrame(summary_rows)

    # Hypothesis tests: Brain vs each other controller
    brain_kills = df[df["controller"] == "brain"]["kills"].values if "brain" in controllers else None

    comparison_lines = ["## Statistical Comparisons (Untrained Brain vs Controls)\n"]
    comparison_lines.append("| Comparison | Brain Mean | Control Mean | Mann-Whitney U | p-value | Cohen's d |")
    comparison_lines.append("|---|---|---|---|---|---|")

    if brain_kills is not None and len(brain_kills) > 0:
        for ctrl in controllers:
            if ctrl == "brain":
                continue
            ctrl_kills = df[df["controller"] == ctrl]["kills"].values
            u_stat, p_val = stats.mannwhitneyu(brain_kills, ctrl_kills, alternative="two-sided")
            d = cohen_d(brain_kills, ctrl_kills)
            comparison_lines.append(
                f"| Brain vs {ctrl} | {np.mean(brain_kills):.2f} | {np.mean(ctrl_kills):.2f} | "
                f"{u_stat:.1f} | {p_val:.4e} | {d:+.2f} |"
            )

    # Hypothesis tests: Trained Brain vs Ablation Controls
    trained_brain_kills = df[df["controller"] == "trained_brain"]["kills"].values if "trained_brain" in controllers else None
    if trained_brain_kills is not None and len(trained_brain_kills) > 0:
        comparison_lines.append("\n## Statistical Comparisons (Trained Brain vs Ablation Controls)\n")
        comparison_lines.append("| Comparison | Trained Brain Mean | Control Mean | Mann-Whitney U | p-value | Cohen's d |")
        comparison_lines.append("|---|---|---|---|---|---|")
        for ctrl in ["trained_shuffled", "trained_random_io", "vnc", "brain", "shuffled", "random_io", "heuristic", "random"]:
            if ctrl in controllers and ctrl != "trained_brain":
                ctrl_kills = df[df["controller"] == ctrl]["kills"].values
                u_stat, p_val = stats.mannwhitneyu(trained_brain_kills, ctrl_kills, alternative="two-sided")
                d = cohen_d(trained_brain_kills, ctrl_kills)
                comparison_lines.append(
                    f"| Trained Brain vs {ctrl} | {np.mean(trained_brain_kills):.2f} | {np.mean(ctrl_kills):.2f} | "
                    f"{u_stat:.1f} | {p_val:.4e} | {d:+.2f} |"
                )

    stats_md = "\n".join(comparison_lines)
    return summary_df, stats_md
