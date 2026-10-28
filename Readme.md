# Fly Brain Arcade 🪰🕹️

> **Space Invaders agent driven by a spiking neural model of the adult *Drosophila melanogaster* brain built from the FlyWire whole-brain connectome (release 783).**

![Fly Brain Arcade Gameplay and Spikes](reports/demo.gif)

---

## 1. Claims Policy

### Allowed Claims:
* *"Behavior emerges from real connectome wiring under a documented, analysis-derived I/O mapping."*
* *"The brain controller outperforms / does not outperform the shuffled-wiring control by X (95% CI)."*

### Prohibited Claims:
* ❌ *"The fly brain learned to play."* (Weights are static structural synapse counts, no plasticity/learning).
* ❌ *"Emulation of a fly brain."* (Weights are synapse counts, signs are predicted neurotransmitters, neuromodulation and gap junctions are absent).
* ❌ Any score claim without comparison to all 4 baseline and control conditions (Random, Heuristic, Shuffled-Wiring, Random-I/O).

---

## 2. Architecture & Conventions

```
FlyWire v783 Parquet/CSV -> Connectome (CSR W, sign, meta)
                                |
                      IOSpec (derived via signed-influence analysis)
                                |
Game Observation -> Encoder -> SpikingBrain.set_poisson_input (Poisson drive)
                               SpikingBrain.step(50 ms) -> SpikeBatch -> Decoder -> Action [left, right, fire]
                                |
                         SpaceInvadersEnv.step(action)
```

* **Sparse Matrix:** $W[\text{post}, \text{pre}]$ stored as `scipy.sparse.csr_matrix` (float32) where $W = \text{sign} \times \text{syn\_count}$. Input current is computed as $W @ \text{spikes}$.
* **Indexing:** Neuron index $i$ corresponds strictly to position in `Connectome.neuron_ids` (sorted FlyWire root IDs).
* **Time Step:** $1\text{ game step} = 50\text{ ms of brain simulation time}$.

---

## 3. Results & Baseline Comparisons

Evaluated across standardized seeds with 95% bootstrap confidence intervals ($n=10{,}000$ resamples) and two-sided Mann–Whitney U tests:

| Controller | Description | Mean Kills | 95% Bootstrap CI | Mean Steps | Accuracy |
|---|---|---|---|---|---|
| **Heuristic** | Upper-bound reference tracking & dodging | 9.67 | [5.00, 12.00] | 442.3 | 35.0% |
| **Random** | Bernoulli action baseline ($p=0.1$) | 2.00 | [0.00, 4.00] | 89.0 | 46.7% |
| **Random-I/O Brain** | Connectome with resampled I/O groups | 0.33 | [0.00, 1.00] | 80.7 | 33.3% |
| **Shuffled Brain** | Degree-preserving edge swapped network | 0.00 | [0.00, 0.00] | 92.7 | 0.0% |
| **Fly Brain (Real)** | Real connectome wiring + derived I/O | 0.00 | [0.00, 0.00] | 70.3 | 0.0% |

### Statistical Comparison (Brain vs Controls)
* **Brain vs Random:** $U = 1.0$, $p = 0.1797$, Cohen's $d = -1.41$
* **Brain vs Heuristic:** $U = 0.0$, $p = 0.0463$, Cohen's $d = -2.71$
* **Brain vs Shuffled-Wiring:** $U = 4.5$, $p = 1.0000$, Cohen's $d = 0.00$
* **Brain vs Random-I/O:** $U = 3.0$, $p = 0.4237$, Cohen's $d = -0.71$

> **Scientific Interpretation:** Under the zero-plasticity baseline with 50 ms time steps, the unadapted fly connectome exhibits a null result against active baselines (Brain $\approx$ Shuffled-Wiring). As mandated by PRD § 2 and § 7, this null result is reported objectively. The structure of the connectome provides lateral steering selectivity (90%+ left/right selectivity under sensory illumination) without achieving autonomous gameplay parity against engineered heuristics.

---

## 4. I/O Selection Justification

* **Sensory Inputs (`enemy_left`, `enemy_right`):** Lateralized visual projection neurons (VPNs, specifically looming/motion responsive types including LC4, LPLC2, LC10, LC16, LC17, LC18).
* **Motor Outputs (`move_left`, `move_right`):** Descending neurons (DNs) selected via signed forward influence analysis ($\max (\text{inf}_{\text{left}} - \text{inf}_{\text{right}})$ for left; mirrored for right). Passes the M3 lateral-selectivity sanity test ($\ge 90\%$ over 20 seeds).
* **Fire Action (`fire`):** Descending neurons activated bilaterally by aligned visual inputs with minimal lateral differential.

---

## 5. How to Reproduce

```bash
# 1. Clone repository
git clone git@github.com:vnparmane/fly_ai.git
cd fly_ai

# 2. Sync dependencies
uv sync

# 3. Fetch connectome data and verify MANIFEST checksums
python -m flybrain.data.fetch

# 4. Run test suite and milestones validation
pytest

# 5. Run full controller evaluation
python -m flybrain.eval.run_eval --controllers random heuristic brain shuffled random_io --seeds 0-29 --out reports/eval_run
```

---

## 6. Citations

1. **Shiu, P. K. et al.** (2024). *A leaky integrate-and-fire computational model based on the connectome of the entire adult Drosophila brain reveals insights into sensorimotor processing.* Nature.
2. **Dorkenwald, S. et al.** (2024). *Neuronal wiring diagram of an adult brain.* Nature (FlyWire Consortium, Release 783).
3. **Stimberg, M., Goodman, D. F., & Brette, R.** (2019). *Brian 2, an intuitive and efficient neural simulator.* eLife.
4. **Kakaria, K. S., & de Bivort, B. L.** (2017). *Ring Attractor Dynamics in the Drosophila Central Complex.* Frontiers in Behavioral Neuroscience.
