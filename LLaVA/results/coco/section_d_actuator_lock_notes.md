# Section III-D Actuator Evidence Lock Notes

This note separates the D-section evidence that is already locked from the part that must be recomputed on the exact current head pool.

## Locked D1: non-selection image routing check

Source:

`LLaVA/results/coco/selected_head_actuator_analysis_l9_l16_k100/head_actuator_analysis_summary.json`

Configuration:

- head pool: exact `global__itext_all__C_toi_HminusG`
- layer window: L9-L16
- top-k: 100 selected heads
- comparison set: 156 non-selected heads inside the same L9-L16 window

Metric:

```text
image mass drop = E_G[M_img] - E_H[M_img]
```

Locked values:

| group | n heads | image mass drop G-H | positive image-drop heads |
|---|---:|---:|---:|
| selected | 100 | 0.026989 | 80 / 100 |
| non-selected | 156 | 0.006629 | 71 / 156 |

Effect:

```text
selected / non-selected image drop = 4.071x
```

Significance:

```text
Mann-Whitney U p = 8.95e-08
P(selected image drop > non-selected image drop) = 0.698
```

This is the cleanest observational D1 statistic because image mass drop is not one of the two direct rank-fusion axes used to select the pool. The text-mass and contrastive-TOI comparisons may still be used as pool characterization, but not as independent actuator evidence.

## Locked D2: exact top-100 causal fragility

Discard the previously discussed D2 numbers:

```text
grounded mean Delta logp      ~= 0.081
hallucinated mean Delta logp  ~= 0.386
hall / grounded ratio         ~= 4.8x
```

Those values were not recomputed on the exact current top-100 L9-L16 pool.

Required D2 specification:

- probe: generic hard/static text-side suppression diagnostic
- head pool: exact top-100 L9-L16 `global__itext_all__C_toi_HminusG`
- not original AD-HH fixed heads
- not the proposed dynamic DEACT decoding run
- base captions: exact greedy n500 sample used by the 0.288 run
- metric:

```text
Delta log p(y_t) = log p_base(y_t) - log p_suppressed(y_t)
```

Server command from `~/Hallucination-Attribution/LLaVA`:

```bash
GPU_ID=6 \
BASE_RESULTS="./results/coco/verify_0288_dynamic_l9_l16_k100_s1_q8_tau0p90_n500_seed42/greedy/captions_eval_results.json" \
PRIOR_PATH="./results/coco/layer_band_dynamic_ablation_head_files/ranked_heads_global__itext_all__C_toi_HminusG_l9_l16.json" \
TOP_K=100 \
MAX_PER_LABEL=200 \
OUTPUT_DIR="./results/coco/section_d_exact_top100_l9_l16_fragility_m200" \
bash bash_scripts/soft_routing/run_section_d_exact_fragility_lock.sh
```

If the base result path differs on the server, set `BASE_RESULTS` to the greedy `captions_eval_results.json` from the exact n500 run that produced:

```text
DEACT dynamic_l9_l16_k100_s1.0_q8.0_tau0.90:
CHAIRs 0.288, CHAIRi 0.07828
```

Locked values from the exact run:

| label | n | mean Delta logp | median Delta logp | q75 Delta logp | q90 Delta logp | positive drop frac | top-1 changed |
|---|---:|---:|---:|---:|---:|---:|---:|
| grounded_object | 200 | -0.003117 | 0.000593 | 0.068176 | 0.197185 | 0.525 | 0.090 |
| hallucinated_object | 200 | 0.148774 | 0.080857 | 0.263504 | 0.574243 | 0.685 | 0.170 |

Do not report a hall/ground ratio for D2 because the grounded mean is near zero and slightly negative. Report the absolute contrast instead:

```text
H-G mean Delta logp gap = 0.148774 - (-0.003117) = 0.151891
top-1 changed gap       = 17.0% - 9.0% = 8.0 percentage points
```

After the D2 run completes, generate the locked D figures and p-values:

```bash
python eval_scripts/soft_routing/build_section_d_locked_figures.py \
  --head-scores-path "./results/coco/layer_band_dynamic_ablation_head_files/ranked_heads_global__itext_all__C_toi_HminusG_l9_l16.json" \
  --fragility-rows-csv "./results/coco/section_d_exact_top100_l9_l16_fragility_m200/static_object_logprob_drop_rows.csv" \
  --top-k 100 \
  --output-dir "./results/coco/section_d_locked_figures"
```

Expected outputs:

```text
section_d_image_mass_drop_locked.svg/png/pdf
section_d_causal_fragility_locked.svg/png/pdf
section_d_locked_summary.json
section_d_locked_summary_flat.csv
```

## D2 and E scale relationship

D2 and E intentionally use different probes unless we explicitly rerun one of them with a matched perturbation strength.

- D2: selected-pool hard/static suppression diagnostic on the exact top-100 pool.
- E: band-wide small/control perturbation used only for layer localization.

Therefore, the absolute `Delta log p` values are not directly comparable. The currently discussed E value for L9-L16 is:

```text
E L9-L16 hallucinated Delta logp = 0.0207
```

If D2 remains near the old value:

```text
D2 hallucinated Delta logp ~= 0.386
```

then D2 is about 18.6x larger than E:

```text
0.386 / 0.0207 ~= 18.6
```

Paper wording must say this explicitly:

> D2 measures the causal fragility of the selected head pool under a full hard/static text-side suppression probe, whereas E uses a small band-wide perturbation only to localize the depth at which hallucination-specific fragility appears. Their absolute magnitudes should not be compared; only the within-probe grounded-vs-hallucinated contrast is interpreted.

If we want to remove this caveat, rerun D2 and E with the same perturbation strength and same head selection policy.

## C dependency

The sentence:

```text
pre-intervention text-reliance remains a weak hallucination classifier (AUC = 0.594)
```

is valid only if Section III-C reports the exact same token-level diagnostic:

Source:

`LLaVA/results/coco/text_image_ratio_diagnostics_top100_l9_l16/text_image_ratio_summary_flat.csv`

Relevant values:

| level | field | label | n | mean | q50 | q75 | q90 |
|---|---|---|---:|---:|---:|---:|---:|
| token | r_img | grounded | 716 | 0.817956 | 0.840702 | 0.895358 | 0.925669 |
| token | r_img | hallucinated | 121 | 0.853427 | 0.868911 | 0.907307 | 0.934510 |

If C uses a different ratio definition or a different head pool, the AUC sentence in D/E must be removed or updated.

## Matched-control status

Matched-control D3 is still not done.

Needed control:

```text
selected top-100 vs random same-size vs text-mass-matched non-selected heads
```

Measured under the same generic suppression probe, with hallucinated and grounded `Delta log p` reported separately.

Until this is run, D can claim:

```text
selected heads show non-circular visual-routing weakening and stronger causal fragility under suppression
```

but should not overclaim:

```text
actuation is independent of text mass
```
