# Head Pool Characterization Notes

Source: `results/coco/layer_band_dynamic_ablation_head_files/ranked_heads_global__itext_all__C_toi_HminusG_l9_l16.json`

Selection: top-100 heads from the L9--L16 ranked pool, with the remaining heads in the same layer window used as the non-selected comparison group.

## Main Finding

This file characterizes what the selected head pool captures. It should not be used by itself as the causal evidence for an actuator claim.

Important circularity caveat: the pool is selected from rank percentiles of text-side mass and contrastive text-over-image score. Therefore, selected-vs-non-selected differences on text-side mass, contrastive TOI score, and log-TOI gap are partly by construction. They are useful for explaining the construction of the pool, not for independently proving that these heads are intervention-relevant.

## Selected vs Non-Selected Heads

| metric | selected top-100 | non-selected |
|---|---:|---:|
| mean text-side mass $I_{text}$ | 0.348 | 0.291 |
| mean positive contrast score $\max(C_{toi},0)$ | 6.311 | 0.066 |
| mean log-TOI gap H-G | 0.299 | -0.235 |
| mean image drop G-H | 0.027 | 0.007 |

Directionality among selected heads:

- positive raw text-over-image gap: 97/100
- positive log text-over-image gap: 97/100
- positive image drop from grounded to hallucinated: 80/100
- positive text-mass gap from grounded to hallucinated: 67/100

Effect sizes selected vs non-selected:

- text-side mass: Cohen's d = 0.378
- positive contrast score: Cohen's d = 0.847
- log TOI gap: Cohen's d = 1.793
- image drop: Cohen's d = 0.582

The text-leverage percentile and contrastive percentile have low linear correlation across the L9--L16 head pool (Pearson r = -0.067). This is best treated as low redundancy between the two ranking axes, not as a causal result.

## Paper-Ready Interpretation

The selected pool is intentionally biased toward heads with high post-image text-side mass and high hallucinated-minus-grounded text-over-image contrast. These properties characterize the two axes used for selection: a text-side leverage axis and a hallucination-state contrast axis. Because these axes are part of the selection rule, they should be presented as a description of the selected pool rather than as independent proof of intervention relevance.

The least circular observation in this table is the image-token mass drop, because image drop is not one of the two main selection axes for this pool. Selected heads show a larger grounded-to-hallucinated image-mass drop (0.027 vs. 0.007), suggesting that hallucinated object generation is accompanied by weakened visual-token routing in this head pool.

To justify the term "actuator", this characterization must be paired with a separate causal diagnostic showing that suppressing these channels changes object-token likelihood. Without that causal panel, the safe claim is only that the selected pool captures text-side leverage and hallucination-state text-over-image bias.

## Caution

This is a head-pool characterization, not a standalone token-level hallucination detector. High text reliance also occurs for grounded objects, so the selected heads should not be described as detecting hallucination. The safe claim is that they are candidate text-side control channels whose text-over-image reliance increases in hallucination states.
