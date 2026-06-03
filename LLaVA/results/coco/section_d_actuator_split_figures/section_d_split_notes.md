# Section III-D Split Figure Notes

The D figure is split into two non-overlapping claims.

## Panel A: image mass drop

Source: `LLaVA/results/coco/selected_head_actuator_analysis_l9_l16_k100/head_actuator_group_summary.csv`

This panel uses only the image-token mass drop:

```text
E_G[M_img] - E_H[M_img]
```

This metric is not one of the two primary selection axes (`I_text` and `C_toi`), so it is the least circular observational statistic for D.

- selected: 0.026989
- non-selected: 0.006629
- ratio: 4.071x

## Panel B: causal fragility

Source: `LLaVA/results/coco/adhh_static_hall_ground_touch_figure/adhh_static_hall_ground_touch_summary.json`

This panel reports:

```text
Delta log p(y_t) = log p_base(y_t) - log p_suppressed(y_t)
```

- grounded mean Delta logp: 0.080929
- hallucinated mean Delta logp: 0.386417
- hallucinated / grounded ratio: 4.775x
- grounded top-1 token change fraction: 0.135
- hallucinated top-1 token change fraction: 0.345

Use this as a causal diagnostic panel. If a stricter current-pool generic suppression run is available, replace `--causal-summary-json` with that source and regenerate the same panel.
