# Reward Weighting Spec (Step 3)

## Scope
- Data scope: `pick_and_place_*`, `egg_carton_to_tray`, `egg_carton_to_tray_cracked`.
- Training mode: offline.
- Goal: reward-weighted flow matching (RWFM) with stable, bounded sample weights.

## Final weighting signal
- Per-sample mixed score:
- `R_mix = beta * z_episode + (1 - beta) * z_chunk`
- `beta = 0.7`

Where:
- `z_episode` is robust-normalized `episode_reward`.
- `z_chunk` is robust-normalized finite-horizon discounted chunk return from step rewards.

## Chunk return definition
- Horizon: `H = 50` (matches action chunk).
- Discount: `gamma = 0.99`.
- For timestep `t` in episode:
- `R_chunk(t) = sum_{k=0..H-1} gamma^k * r_{t+k}` (truncate at episode end).

## Robust normalization (anti-scale-bias across datasets)
- Group key: **task string** (from `meta/tasks.parquet`).
- For each group and each signal (`episode_reward`, `chunk_return`):
- `median_g = median(x_g)`
- `mad_g = median(|x_g - median_g|)`
- `scale_g = max(1.4826 * mad_g, eps)` with `eps = 1e-6`
- If `scale_g < eps`, fallback to `std`; if still too small, fallback to `1.0`.
- Robust z-score:
- `z = (x - median_g) / scale_g`
- Clipping:
- `z_episode = clip(z_episode, -6, 6)`
- `z_chunk = clip(z_chunk, -6, 6)`
- `R_mix = clip(R_mix, -6, 6)`

## Weight formula (RWFM)
- Advantage proxy: `A = R_mix`
- Raw weight: `w_raw = exp(alpha * A)`
- Chosen (recalibrated on 2026-02-20): `alpha = 0.25`
- Hard clip: `w_clip = clip(w_raw, w_min, w_max)` with:
- `w_min = 0.25`
- `w_max = 4.0`
- Batch normalization:
- `w = w_clip / mean_batch(w_clip)`

This `w` is used to weight per-sample FM loss.

## Diagnostics and stability check
- Stats files:
- Clean cycle (no cracked): `crab/training/docs/reward_weighting_stats_cycle1_clean.json`
- Controlled cracked factor: `crab/training/docs/reward_weighting_stats_cycle1_with_cracked.json`
- Generated with:
```bash
python crab/training/scripts/compute_reward_weighting_stats.py ^
  --root c:\\Users\\user\\Downloads\\vla ^
  --group-key task --horizon 50 --gamma 0.99 ^
  --mix-beta 0.7 --alpha 0.25 --w-min 0.25 --w-max 4.0 ^
  --batch-size 16 --sim-batches 4000
```

Observed for recalibrated config (`alpha=0.25`):
- Clean cycle:
- `clip_low_frac = 0.0160`
- `clip_high_frac = 0.0`
- `ESS_mean = 0.8310`
- `ESS_p10 = 0.7184`
- With cracked:
- `clip_low_frac = 0.0329`
- `clip_high_frac = 0.0`
- `ESS_mean = 0.8012`
- `ESS_p10 = 0.6817`

Interpretation:
- No overflow/high-end collapse (`clip_high_frac=0`).
- Low-end clipping is moderate and controlled.
- Effective sample size is high enough to avoid batch domination.

## Acceptance criterion (Step 3)
- Weighting is accepted if:
- `clip_high_frac <= 0.01`
- `clip_low_frac <= 0.25`
- `ESS_p10 >= 0.65` (with target batch mix and batch size)

Current recalibrated config passes all three checks for both clean and cracked-controlled runs.

## Implementation note for next step
- Precompute and store group robust stats per run composition (clean vs cracked-controlled).
- In training code, compute per-sample `w` exactly by this spec before loss reduction.
