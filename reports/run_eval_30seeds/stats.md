# Fly Brain Arcade — Evaluation Statistics

## Overall Performance Summary

| controller   |   n_episodes |   kills_mean | kills_ci95     |   steps_mean | steps_ci95     | accuracy   |
|:-------------|-------------:|-------------:|:---------------|-------------:|:---------------|:-----------|
| random       |           30 |     2        | [1.47, 2.57]   |      123.433 | [87.7, 170.2]  | 36.1%      |
| heuristic    |           30 |    12.4667   | [11.57, 13.33] |      551.667 | [504.0, 589.7] | 33.5%      |
| brain        |           30 |     0        | [0.00, 0.00]   |      103.2   | [79.8, 135.9]  | 0.0%       |
| shuffled     |           30 |     0        | [0.00, 0.00]   |      102.733 | [85.5, 121.8]  | 0.0%       |
| random_io    |           30 |     0.533333 | [0.23, 0.87]   |      134.2   | [94.5, 183.9]  | 27.3%      |

## Statistical Comparisons (Brain vs Controls)

| Comparison | Brain Mean | Control Mean | Mann-Whitney U | p-value | Cohen's d |
|---|---|---|---|---|---|
| Brain vs random | 0.00 | 2.00 | 60.0 | 1.6244e-10 | -1.82 |
| Brain vs heuristic | 0.00 | 12.47 | 0.0 | 7.5762e-13 | -7.05 |
| Brain vs shuffled | 0.00 | 0.00 | 450.0 | nan | +0.00 |
| Brain vs random_io | 0.00 | 0.53 | 300.0 | 6.5166e-04 | -0.84 |