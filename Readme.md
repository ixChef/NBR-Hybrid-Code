# NBR-Hybrid-Code

This repository contains the **hybrid orchestration, preprocessing adapters, score export pipeline, alignment checks, and score-level fusion code** used in my FYP on **temporal next-basket recommendation**.

The project compares and combines two temporal next-basket recommendation families:

- **TIFU-TD** from the `time_dependent_nbr` codebase
- **TAIW** from the `time_aware_item_weighting` codebase

The final project focuses only on the following datasets:

- **Dunnhumby**
- **Ta-Feng**

The main goal of this repository is to:

1. reproduce and compare both model families under a controlled setup,
2. treat preprocessing/profile construction as an experimental variable,
3. build an aligned **score-level hybrid ensemble**.

---

## Important note

This repository **only contains the hybrid-related code**.  
It does **not** contain the full original TIFU-TD and TAIW repositories.

To run this project properly, you must also obtain:

- the original **`time_dependent_nbr`** repository
- the original **`time_aware_item_weighting`** repository
- the raw **Dunnhumby** and **Ta-Feng** datasets

---

## Expected folder structure

After cloning everything, the repo root should look like this:

```text
NBR-Hybrid-Code/
├── hybrid_nbr/
│   ├── analysis/
│   ├── configs/
│   ├── data/
│   ├── preprocessing/
│   ├── results/
│   └── runners/
├── time_dependent_nbr/
├── time_aware_item_weighting/
├── requirements.txt
├── flake.nix
└── flake.lock
```

The two external repositories must be placed **next to** `hybrid_nbr`, not inside it.

---

## Required external repositories

You must clone or otherwise place the following repositories in the root of this project:

- `time_dependent_nbr`
- `time_aware_item_weighting`

These are required because the hybrid runners import code directly from them.

---

## Required datasets

Only these two datasets are used in the final project:

- **Dunnhumby**
- **Ta-Feng**

You must download them separately and place the raw files in the correct locations expected by the original repositories.

The original experiments used public versions of:

- **Dunnhumby**: `frtgnn/dunnhumby-the-complete-journey`
- **Ta-Feng**: `chiranjivdas09/ta-feng-grocery-dataset`

---

## Environment setup

From the **repo root**:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you are using Nix, you can also use the provided `flake.nix`.

---

## Where to run commands

All project commands should be run from inside the `hybrid_nbr` folder.

So the standard start is:

```bash
cd hybrid_nbr
source ../.venv/bin/activate
```

If your virtual environment is located elsewhere, activate it accordingly.

---

## Profiles used in the project

This project uses three profile types:

- `tifu_profile`
- `taiw_profile`
- `tifu_hybrid_profile`

### Meaning of each profile

- `tifu_profile`: native corrected TIFU preprocessing
- `taiw_profile`: native corrected TAIW preprocessing
- `tifu_hybrid_profile`: a special hybrid-only TAIW-compatible corpus generated from native TIFU splits for aligned score fusion

---

## What the hybrid is

The hybrid is a **score-level weighted linear ensemble**:

\[
H(u,i)=\alpha S^{TAIW}_{norm}(u,i) + (1-\alpha) S^{TIFU}_{norm}(u,i)
\]

Where:

- `u` is the user
- `i` is the item
- `S_TAIW_norm(u,i)` is the normalised TAIW score
- `S_TIFU_norm(u,i)` is the normalised TIFU score
- `alpha` is tuned on the development set

This means the final item ranking is produced by combining the normalised output scores of both models.

---

## Baseline matrix

The baseline comparison covers:

1. **TIFU on TIFU**
2. **TAIW on TAIW**
3. **TAIW on TIFU**
4. **TIFU on TAIW**

This allows us to separate:

- model-family effects
- preprocessing/profile effects
- model–profile interaction effects

---

## Main scripts

### Baseline execution
- `runners/run_tifu_on_profile.py`
- `runners/taiw_route_a.py`

### Preprocessing and profile conversion
- `preprocessing/taiw_to_tifu_adapter.py`
- `preprocessing/tifu_to_taiw_adapter.py`
- `preprocessing/tifu_to_taiw_hybrid_adapter.py`

### Hybrid pipeline
- `runners/fusion_common.py`
- `runners/run_taiw_export_scores.py`
- `runners/run_tifu_export_scores.py`
- `runners/check_score_alignment.py`
- `runners/run_score_fusion.py`
- `runners/run_score_fusion_multiseed.py`

### Analysis / plotting
- `analysis/generate_baseline_plots.py`

---

## How to run the code

## 1. Move into the hybrid project

```bash
cd hybrid_nbr
source ../.venv/bin/activate
```

---

## 2. Run baseline experiments

### Native TIFU on TIFU

```bash
python runners/run_tifu_on_profile.py \
  --dataset <dunnhumby|tafeng> \
  --profile-root ./data/tifu_profile \
  --alias-prefix tifu_profile \
  --model tifuknn_time_days_next_ts \
  --num-trials 300 \
  --batch-size 20000
```

### Native TAIW on TAIW

```bash
python runners/taiw_route_a.py tune \
  --dataset <dunnhumby|tafeng> \
  --profile taiw_profile \
  --search-space knn_only \
  --tune-seed 10

python runners/taiw_route_a.py status \
  --dataset <dunnhumby|tafeng> \
  --profile taiw_profile \
  --search-space knn_only \
  --tune-seed 10

python runners/taiw_route_a.py eval \
  --dataset <dunnhumby|tafeng> \
  --profile taiw_profile \
  --search-space knn_only \
  --tune-seed 10 \
  --seeds 0 1 2 3 4 5 6 7 8 9
```

### Cross TAIW on TIFU

```bash
python runners/taiw_route_a.py tune \
  --dataset <dunnhumby|tafeng> \
  --profile tifu_profile \
  --profile-root-override ./data/tifu_profile_taiw \
  --search-space knn_only \
  --tune-seed 10

python runners/taiw_route_a.py eval \
  --dataset <dunnhumby|tafeng> \
  --profile tifu_profile \
  --profile-root-override ./data/tifu_profile_taiw \
  --search-space knn_only \
  --tune-seed 10 \
  --seeds 0 1 2 3 4 5 6 7 8 9
```

### Cross TIFU on TAIW

```bash
python runners/run_tifu_on_profile.py \
  --dataset <dunnhumby|tafeng> \
  --profile-root ./data/taiw_profile_tifu \
  --alias-prefix taiw_profile \
  --model tifuknn_time_days_next_ts \
  --num-trials 300 \
  --batch-size 20000
```

---

## 3. Generate baseline plots

```bash
python analysis/generate_baseline_plots.py
```

Outputs are saved under:

```text
results/baseline_plots/
```

---

## 4. Run the hybrid on TAIW profile

### Export TAIW scores

```bash
for seed in 0 1 2 3 4 5 6 7 8 9; do
  python runners/run_taiw_export_scores.py \
    --dataset dunnhumby \
    --profile taiw_profile \
    --search-space knn_only \
    --tune-seed 10 \
    --eval-seed "$seed"
done
```

### Export TIFU-on-TAIW scores

```bash
python runners/run_tifu_export_scores.py \
  --dataset dunnhumby \
  --profile-root ./data/taiw_profile_tifu \
  --alias-prefix taiw_profile \
  --model tifuknn_time_days_next_ts \
  --batch-size 20000
```

### Check alignment

```bash
python runners/check_score_alignment.py \
  --left-dir ./results/score_exports/taiw/knn_only/taiw_profile/dunnhumby/seed_0 \
  --right-dir ./results/score_exports/tifu/taiw_profile/dunnhumby/tifuknn_time_days_next_ts \
  --output-json ./results/score_exports/alignment_dunnhumby_taiw_profile.json
```

### Run multi-seed fusion

```bash
python runners/run_score_fusion_multiseed.py \
  --left-root ./results/score_exports/taiw/knn_only/taiw_profile/dunnhumby \
  --right-dir ./results/score_exports/tifu/taiw_profile/dunnhumby/tifuknn_time_days_next_ts \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --topk 10 \
  --alpha-step 0.05 \
  --output-dir ./results/score_fusion/dunnhumby/taiw_profile/taiw_plus_tifu_multiseed
```

Repeat the same pattern for `tafeng`.

---

## 5. Run the hybrid on TIFU profile

### Build hybrid-aligned TAIW corpus

```bash
python preprocessing/tifu_to_taiw_hybrid_adapter.py --dataset dunnhumby
python preprocessing/tifu_to_taiw_hybrid_adapter.py --dataset tafeng
```

### Tune TAIW on the new hybrid-aligned profile

```bash
python runners/taiw_route_a.py tune \
  --dataset dunnhumby \
  --profile tifu_hybrid_profile \
  --search-space knn_only \
  --tune-seed 10
```

Check progress:

```bash
python runners/taiw_route_a.py status \
  --dataset dunnhumby \
  --profile tifu_hybrid_profile \
  --search-space knn_only \
  --tune-seed 10
```

### Export TAIW scores on `tifu_hybrid_profile`

```bash
for seed in 0 1 2 3 4 5 6 7 8 9; do
  python runners/run_taiw_export_scores.py \
    --dataset dunnhumby \
    --profile tifu_hybrid_profile \
    --search-space knn_only \
    --tune-seed 10 \
    --eval-seed "$seed"
done
```

### Export native TIFU-on-TIFU scores

```bash
python runners/run_tifu_export_scores.py \
  --dataset dunnhumby \
  --profile-root ./data/tifu_profile \
  --alias-prefix tifu_profile \
  --model tifuknn_time_days_next_ts \
  --batch-size 20000
```

### Check alignment

```bash
python runners/check_score_alignment.py \
  --left-dir ./results/score_exports/taiw/knn_only/tifu_hybrid_profile/dunnhumby/seed_0 \
  --right-dir ./results/score_exports/tifu/tifu_profile/dunnhumby/tifuknn_time_days_next_ts \
  --output-json ./results/score_exports/alignment_dunnhumby_tifu_hybrid_profile.json
```

### Run multi-seed fusion

```bash
python runners/run_score_fusion_multiseed.py \
  --left-root ./results/score_exports/taiw/knn_only/tifu_hybrid_profile/dunnhumby \
  --right-dir ./results/score_exports/tifu/tifu_profile/dunnhumby/tifuknn_time_days_next_ts \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --topk 10 \
  --alpha-step 0.05 \
  --output-dir ./results/score_fusion/dunnhumby/tifu_profile/taiw_plus_tifu_multiseed
```

Repeat the same pattern for `tafeng`.

---

## Checking whether all TAIW export seeds are done

Example check for seeds `0..9`:

```bash
for seed in 0 1 2 3 4 5 6 7 8 9; do
  dir="./results/score_exports/taiw/knn_only/tifu_hybrid_profile/tafeng/seed_${seed}"

  if [ -f "$dir/meta.json" ] \
     && [ -f "$dir/item_ids.npy" ] \
     && [ -f "$dir/scores_dev.npy" ] \
     && [ -f "$dir/scores_test.npy" ] \
     && [ -f "$dir/user_ids_dev.npy" ] \
     && [ -f "$dir/user_ids_test.npy" ] \
     && [ -f "$dir/targets_dev.json" ] \
     && [ -f "$dir/targets_test.json" ]; then
    echo "seed ${seed}: OK"
  else
    echo "seed ${seed}: MISSING OR INCOMPLETE"
  fi
done
```

---

## Final hybrid results

| Dataset | Hybrid profile | Precision@10 | Recall@10 | NDCG@10 | Mean best alpha |
|---|---:|---:|---:|---:|---:|
| Dunnhumby | TAIW profile | 0.121740 | 0.178464 | 0.208989 | 0.50 |
| Ta-Feng | TAIW profile | 0.067265 | 0.164900 | 0.152695 | 0.32 |
| Dunnhumby | TIFU profile | 0.123191 | 0.243646 | 0.240541 | 0.62 |
| Ta-Feng | TIFU profile | 0.067127 | 0.158771 | 0.145364 | 0.06 |

---

## Summary

This repository implements a **hybrid ensemble framework** for temporal next-basket recommendation by combining TIFU-TD and TAIW at the score level.

The core pipeline is:

1. run or reconstruct the two baseline recommenders,
2. export aligned dev/test score matrices,
3. verify alignment explicitly,
4. normalise both score spaces per user,
5. fuse them with weighted linear combination,
6. tune `alpha` on the development set,
7. evaluate the final hybrid on the test set.

This repository is intended to support the **hybrid experiments only**, not to replace the original TIFU-TD or TAIW repositories.
