# Chart-Consistent Mixture-of-Experts Policies for Sample-Efficient Continuous Control
This repository contains a clean PyTorch implementation of the **Chart-Consistent Mixture-of-Experts PPO (CCMoE-PPO)** and the baselines used in our HalfCheetah-v5 experiments:

- **CCMoE-PPO (proposed)** – chart-consistent mixture-of-experts policy with:
  - Shared encoder features
  - Euclidean chart cover in feature space
  - Masked, locality-aware gating
  - Overlap-conditioned consistency on chart intersections
  - Optional gradient balancing for the overlap loss
- **Single-Gaussian PPO** – standard diagonal Gaussian actor–critic baseline
- **MoE-PPO** – mixture-of-experts PPO with global gate (no chart structure)
- **Graph-Laplacian PPO** – hard Voronoi partition + parameter-space Laplacian penalty

> ⚠️ **Important**: To fully understand the design choices, losses, and hyperparameters in this code, you **should read the corresponding CCMoE paper**. The paper explains the geometry, consistency constraints, and the theoretical motivation that this implementation follows.

---

## 1. File Overview

The main script (you can rename it as you wish, e.g. `ccmoe_halfcheetah.py`) implements:

- **Algorithms**
  - `SingleGaussianPPO` – vanilla PPO
  - `MoEPPO` – mixture-of-experts PPO
  - `GraphLaplacianPPO` – PPO + graph Laplacian regularization on expert heads
  - `CCMoE` – proposed chart-consistent mixture-of-experts policy

- **Key components**
  - `MLP` – shared encoder \( \phi_\theta(o) \)
  - `ChartCover` – maintains the chart centers and radii in feature space
  - `Gate` – gating network producing expert logits
  - `LocalGaussianHead` – per-chart Gaussian policies
  - `SquashedDiagGaussian` – tanh-squashed diagonal Gaussian policy
  - `RolloutBuffer` – storage for PPO rollouts with GAE(λ)
  - `RunLogger` – CSV + JSON logging for each run

- **Experiment orchestration**
  - `run_one(...)` – run one training configuration
  - `run_suite(...)` – run full grid of CCMoE + baselines
  - `aggregate_summary()` – aggregate all summaries into `aggregate_summary.json` and `.csv`
  - Command-line interface for configuring runs

All experiment outputs are stored under:

```text
CCMoE_HalfCheetah/
  ├── ccmoe/
  ├── ppo/
  ├── ppo_glap/
  ├── moe/
  ├── master_index.csv
  ├── aggregate_summary.csv
  └── aggregate_summary.json
```
- Each algorithm subfolder contains one directory per run with:
  - `config.json` – serialized `TrainConfig`
  - `summary.json` – final metrics (AUC, final return, time-to-threshold, etc.)
  - `episode.csv` – per-episode returns vs. environment steps
  - `train.csv` – PPO + regularizer diagnostics
  - `checkpoints/` – model checkpoints
The main script (e.g. ccmoe_halfcheetah.py) implements:

- Algorithms:
  - `CCMoE` – proposed chart-consistent mixture-of-experts policy
  - `SingleGaussianPPO` – standard PPO
  - `MoEPPO` – mixture-of-experts PPO baseline
  - `GraphLaplacianPPO` – PPO + graph Laplacian baseline
- Core components:
  - `MLP` – shared encoder
  - `ChartCover` – maintains chart centers and radius in feature space
  - `Gate` – gating network producing expert logits
  - `LocalGaussianHead` – per-chart Gaussian policies
  - `SquashedDiagGaussian` – tanh-squashed diagonal Gaussian
  - `RolloutBuffer` – PPO rollouts with GAE(λ)
  - `RunLogger` – CSV + JSON logging
- Experiment orchestration:
  - `run_one(...)` – run one training configuration
  - `run_suite(...)` – run CCMoE + baselines over a hyperparameter grid
  - `aggregate_summary(...)` – aggregate results across runs
## 2. Running Experiments
 - **Run the full suite (CCMoE + baselines + grid over hyperparameters)**
   This reproduces the grid described in the paper for HalfCheetah-v5 (you may adjust the ranges to exactly match the paper):
```text
python ccmoe_halfcheetah.py \
  --run_all \
  --steps 800000 \
  --seeds 0,1,2 \
  --m_list 2,4,8 \
  --r_list 1.5,2.0,2.5 \
  --lambda_list 0.0,0.01,0.05 \
  --restrictions identity,learned
```
What this does:
 -  Runs CCMoE-PPO over a grid of:
    - Number of charts `m`
    - Chart radius `r`
    - Overlap penalty weight `λ`
    - Restriction type (`identity` or `learned`)
 - Runs PPO, MoE-PPO, and Graph-Laplacian PPO baselines with compatible settings.
 - Logs the results under CCMoE_HalfCheetah/ and aggregates metrics.
You can run just CCMoE or any single baseline:
**CCMoE-PPO (proposed)**
```text
python ccmoe_halfcheetah.py \
  --algo ccmoe \
  --seeds 0 \
  --steps 800000 \
  --m_list 4 \
  --r_list 2.0 \
  --lambda_list 0.01 \
  --restrictions identity \
  --gate_ent_coef 0.0 \
  --grad_balance_alpha 0.0
```
**Single-Gaussian PPO**
```text
python ccmoe_halfcheetah.py \
  --algo ppo \
  --seeds 0 \
  --steps 800000
```
**MoE-PPO baseline**
```text
python ccmoe_halfcheetah.py \
  --algo moe \
  --m_list 4 \
  --seeds 0 \
  --steps 800000
```
**Graph-Laplacian PPO baseline**
```text
python ccmoe_halfcheetah.py \
  --algo ppo_glap \
  --m_list 4 \
  --r_list 2.0 \
  --k_lap 2 \
  --lap_scale 1e-4 \
  --seeds 0 \
  --steps 800000
```
> 💡 **Note**: For single-run mode (`--run_all off`), the script reads only the first value of each `*_list` argument (e.g. the first element of `--m_list`, `--r_list`, `--lambda_list`).

## 3. Important Hyperparameters (see the paper for details)
The following options are exposed via TrainConfig and the CLI.
The paper is the source of truth for how they should be set to reproduce the reported results.
**Algorithm**
  - `--algo` in `{ccmoe, ppo, ppo_glap, moe}`
**Charts / Cover**
  - `--m_list` – number of charts `m`
  - `--r_list` – chart radius `r` in whitened feature space
  - `--restrictions` – `"identity"` or `"learned"` restriction maps
**CCMoE Overlap / Regularization**
  - `--lambda_list` – maximum overlap penalty `λ_max` (or fixed `λ` if gradient balancing is off)
  - `--lam_min` – lower bound `λ_min` during gradient balancing
  - `--grad_balance_alpha` – gradient balancing coefficient `α`
  - `--gate_ent_coef` – gate entropy coefficient `η`
**PPO Core**
  - `--steps` – total environment steps per run
  - `--update_freq` – steps per PPO update (rollout batch size)
  - `--epochs` – PPO epochs per update
  - `--mb_size` – mini-batch size within PPO update
  - `--lr` – learning rate
  - `--vf_coef`, `--ent_coef` – value loss and policy entropy weights
  - `--clip_ratio` – PPO clipping range
**Graph-Laplacian baseline**
  - `--k_lap` – `k` for k-NN graph on chart centers
  - `--lap_scale` – Laplacian penalty weight
To understand why these hyperparameters exist and how they relate to CCMoE’s manifold and overlap structure, please read the paper. The paper explains the theoretical motivation and the recommended grids.

 ## 4. Reproducing the Paper
To reproduce the experimental results:
 1. Read the CCMoE paper carefully. It explains:
    - The geometry of the chart cover and restrictions
    - The exact form of the consistency loss and gradient balancing
    - The hyperparameter grids and training protocol
    - Implementation details that are not obvious from code alone
 2. Match the hyperparameters:
    - Set `--m_list, --r_list, --lambda_list, --gate_ent_coef, --grad_balance_alpha, etc.`, according to the experimental section of the paper.
 3. Run multiple seeds:
    - Use the same number of seeds as in the paper (e.g. `--seeds 0,1,2` or more) and aggregate the results.
 4. Post-process logs. Use:
    - episode.csv
    - train.csv
    - aggregate_summary.json
    to build your evaluation tables and plots.
 > **Again**: reading the paper is necessary if you want to fully understand the process and reproduce the reported results.

 ## 5. Citing
 **If you use this code in your research, please cite the CCMoE paper** 
```text
@software{messlem_2025,
	author       = {Messlem, Abdelkader and
	Messlem, Youcef},
	title        = {Chart-Consistent Mixture-of-Experts PPO (CCMoE-
	PPO)
	},
	month        = nov,
	year         = 2025,
	publisher    = {Zenodo},
	version      = {1.0.0},
	doi          = {10.5281/zenodo.17605495},
	url          = {https://doi.org/10.5281/zenodo.17605495},
}
```
