# Section III-D/F Revised Analysis Notes

This note rewrites the analysis logic for:

- **III-D. Hallucination Heads as Text-Side Actuators**
- **III-F. Static Suppression Is an Over-Broad Intervention**

The main correction is conceptual. The selected head pool must not be presented as a hallucination detector, and the selected-vs-non-selected feature plot must not be used as causal proof. The clean story is:

```text
D: selected heads are characterized by text-side leverage and hallucination-state text-over-image bias, but that characterization is partly by construction; the non-circular observation is stronger image-token mass drop, and an actuator claim needs a separate causal diagnostic.

F: static hard suppression is useful but broad; at the top-100 L9-L16 operating point, grounded object head-steps still cross the hard suppression threshold at a high absolute rate, and grounded object nodes are measurably reduced or disappeared.
```

---

## Data Sources

### D Characterization

- `LLaVA/results/coco/selected_head_actuator_analysis_l9_l16_k100/head_actuator_group_summary.csv`
- `LLaVA/results/coco/selected_head_actuator_analysis_l9_l16_k100/head_actuator_direction_counts.csv`
- `LLaVA/results/coco/selected_head_actuator_analysis_l9_l16_k100/head_actuator_paper_notes.md`
- `LLaVA/results/coco/selected_head_actuator_analysis_l9_l16_k100/selected_vs_nonselected_actuator_metrics.svg`

The figure title has been downgraded to **Selected head pool characterization**. It should explain what the selected pool captures, not prove intervention relevance.

### F Static Over-Broadness

- `LLaVA/results/coco/static_overbroad_top100_l9_l16/static_overbroad_top100.svg`
- `LLaVA/results/coco/static_overbroad_top100_l9_l16/static_overbroad_top100_summary.json`
- `LLaVA/results/coco/static_overbroad_top100_l9_l16/static_overbroad_top100_trigger_summary.csv`
- `LLaVA/results/coco/adhh_hard_tau0p4_removal_loss_disappear/adhh_removal_loss_summary.json`

The main F figure now uses the **top-100 L9-L16** head pool, matching the current method operating point more closely than the previous top-150 version.

---

# III-D. Hallucination Heads as Text-Side Actuators

## What D Can Safely Claim

The selected pool is built from two offline axes:

```text
S(l,h) = 1/2 P(I_text(l,h)) + 1/2 P(C_toi(l,h))
```

Therefore, the following selected-vs-non-selected differences are not independent evidence:

- text-side mass
- contrastive TOI score
- log-TOI gap, which is a log-ratio version of the same contrastive axis

These metrics are still useful, but only as **pool characterization**:

| metric | selected top-100 | non-selected | status |
|---|---:|---:|---|
| text-side mass `Itext_all` | 0.348 | 0.291 | by construction |
| contrastive TOI score `C_toi H-G` | 6.311 | 0.066 | by construction |
| log-TOI gap H-G | 0.299 | -0.235 | same contrastive axis |
| image mass drop G-H | 0.027 | 0.0066 | least circular observation |

The non-circular part is the image-token mass drop. Since image drop is not the primary selection score, it can be used as observational evidence that selected heads are associated with weakened visual-token routing during hallucinated object generation.

## How to Phrase Figure 1

Do not caption Figure 1 as proving actuator behavior.

Use this framing:

> Figure 1 characterizes the selected head pool. The first three panels reflect the offline axes used to construct the pool: post-image text-side mass and hallucinated-minus-grounded text-over-image contrast. These panels explain what the selection rule captures, rather than independently validating the actuator claim. The image-mass-drop panel is less circular and shows that selected heads are also associated with weaker visual-token routing in hallucinated object states.

Avoid:

```text
These four metrics support the actuator interpretation.
Selected heads are intervention-relevant channels.
```

Use:

```text
These metrics characterize the selected pool.
The least circular observation is the image-token mass drop.
The actuator interpretation requires causal evidence.
```

## Required Causal Panel for D

The word **actuator** implies a control point: suppressing the channel should change object-token likelihood. Observational attention statistics alone do not prove this.

The causal D panel should therefore report a generic suppression diagnostic on the current selected pool, not AD-HH-specific settings. The target measurement should be:

```text
Delta log p(y_t) = log p_base(y_t) - log p_suppressed(y_t)
```

reported separately for hallucinated and grounded object tokens.

Ideal D causal result:

| bucket | desired pattern |
|---|---|
| hallucinated object | larger Delta log p |
| grounded object | smaller but nonzero Delta log p |
| hallucinated / grounded ratio | greater than 1 |

This is the evidence that turns "text-heavy, contrastive heads" into "actuator channels."

## Verified Local Diagnostic Available Now

The local files include a single-head zero-ablation teacher diagnostic:

- `LLaVA/results/coco/static_suppression_fragility_analysis_l9_l16_k150/single_head_teacher_effect_summary.csv`

It is useful as mechanistic support but should not be treated as the final D causal panel because it is not the exact current selected-pool generic suppression test.

Verified values:

| label family | mean causal effect | positive-effect fraction | mean positive Delta log p |
|---|---:|---:|---:|
| hallucinated | 0.00354 | 54.3% | 0.0263 |
| kept grounded | 0.00035 | 50.2% | 0.0172 |
| lost grounded | 0.00900 | 55.5% | 0.0329 |

Interpretation:

> Single-head ablation can reduce object log-probability in hallucinated and grounded cases. This supports the idea that these heads are control channels, but it should be cited as a diagnostic, not as the final actuator proof.

The previously discussed numbers, such as "4.13x hallucinated fragility" or "Spearman 0.687", should only be placed in the paper if the exact source file is attached and the metric definition is made explicit. I did not find a local source file that cleanly verifies those exact values in the current workspace.

## C vs D Consistency Sentence

Add this sentence to close the apparent contradiction between detector failure and contrastive head bias:

> The selected heads can show a larger group-level hallucinated-minus-grounded text-over-image bias while still failing as token-level hallucination detectors, because the grounded and hallucinated distributions overlap substantially. Group-average bias identifies a useful head-pool prior; it does not provide reliable online token classification.

## Paper Paragraph Draft for D

The selected head pool should first be interpreted as a characterization of two offline axes rather than as causal evidence. By construction, the pool has elevated post-image text-side mass and elevated hallucinated-minus-grounded text-over-image contrast, since these quantities define the fused score. This explains what the pool captures: text-side leverage and hallucination-state text-over-image bias. The less circular observation is that selected heads also show a larger image-token mass drop from grounded to hallucinated object steps (0.027 vs. 0.0066), suggesting that hallucinated object generation in this pool is accompanied by weakened visual-token routing. However, these attention statistics alone do not prove that the heads are actuators. The actuator claim requires a separate counterfactual diagnostic showing that suppressing these channels changes object-token likelihood. Thus, the selected heads are not hallucination detectors; they are candidate text-side control channels whose group-level bias must be paired with causal suppression evidence.

---

# III-F. Static Suppression Is an Over-Broad Intervention

## Correct F Claim

F should not argue that static suppression fails. It should argue:

> Static hard suppression is effective but too broad as an intervention unit. Even at the top-100 L9-L16 operating point, grounded object head-steps cross the hard threshold at a high absolute rate, so a fixed hard gate exposes normal grounded object generation to suppression.

The key is not that the hallucinated and grounded rates are identical. They are not identical at top-100. The key is that grounded exposure is high.

## Top-100 Static Trigger Rates

Static hard gate:

```text
trigger if text_mass >= 0.4
```

Top-100 L9-L16 selected heads:

| object state | triggered head-steps | total head-steps | trigger rate |
|---|---:|---:|---:|
| hallucinated object | 5,222 | 12,100 | 43.16% |
| grounded object | 28,089 | 71,600 | 39.23% |

This replaces the previous top-150 headline of 45.85% vs. 44.63%.

Correct phrasing:

> The rates are not identical, but grounded object steps still cross the hard threshold in 39.2% of selected head-steps, close to the hallucinated rate of 43.2%. Thus, the hard gate exposes grounded object generation to suppression at a high absolute frequency.

Avoid:

```text
The trigger rates are nearly identical.
```

## Ratio-Axis Check

The static hard baseline uses raw text mass, while the later dynamic method uses text-over-image ratio. To reduce axis-confound concerns, the same top-100 trace was also summarized with:

```text
trigger if r = T/(T+I) >= 0.9
```

| object state | ratio-triggered head-steps | total head-steps | trigger rate |
|---|---:|---:|---:|
| hallucinated object | 5,419 | 12,100 | 44.79% |
| grounded object | 27,500 | 71,600 | 38.41% |

The same qualitative point remains on the ratio axis: grounded object steps are still heavily exposed.

## Grounded Object Outcome

Hard suppression changes grounded object content:

| grounded object outcome | count | rate |
|---|---:|---:|
| grounded object nodes | 1,299 | - |
| preserved | 936 | 72.06% |
| partially reduced | 175 | 13.47% |
| disappeared | 188 | 14.47% |
| reduced total | 363 | 27.94% |

Use "partially reduced" consistently, not "reduced" when referring only to the partial category.

Strongest evidence:

```text
14.5% of grounded object nodes disappear entirely.
```

Then the total grounded reduction:

```text
27.9% are reduced = 13.5% partially reduced + 14.5% disappeared.
```

## Paper Paragraph Draft for F

Static hard suppression is a useful but over-broad intervention. Under the top-100 L9-L16 head pool, the hard rule `text_mass >= 0.4` triggers on 43.2% of hallucinated object head-steps, but it also triggers on 39.2% of grounded object head-steps. The rates are not identical, but the grounded exposure is high in absolute terms. A ratio-axis check gives the same qualitative result: with `T/(T+I) >= 0.9`, grounded object head-steps are still triggered 38.4% of the time, compared with 44.8% for hallucinated object head-steps. This broad exposure materializes in the generated captions. In the hard-suppressed outputs, 27.9% of grounded object nodes are reduced, including 13.5% that are partially reduced and 14.5% that disappear entirely. Static suppression therefore can lower hallucination, but it does so by applying a coarse hard gate to channels that also support ordinary grounded object realization.

## D to F Bridge

Use this bridge after D includes causal fragility evidence:

> Even if hallucinated objects are more fragile under suppression, static triggering remains broad: grounded object steps cross the hard threshold in roughly 39% of selected head-steps. This explains why a useful actuator can still damage grounded content when controlled by a fixed hard gate. The problem is not the existence of text-side leverage; the problem is that static suppression cannot target that leverage selectively enough.

## Do Not Close the Loop in Section III

Do not claim here that DEACT solves the grounded damage unless the same node-level grounded-preservation metric has been computed for DEACT.

Use:

```text
The quantitative comparison with the proposed dynamic method is deferred to Section V.
```

Avoid:

```text
DEACT fixes this grounded loss.
```

---

# Figure Updates

## Revised Figure 1

Use:

- `LLaVA/results/coco/selected_head_actuator_analysis_l9_l16_k100/selected_vs_nonselected_actuator_metrics.svg`

Caption stance:

```text
Selected pool characterization, not causal proof.
```

## Revised Figure 2

Use:

- `LLaVA/results/coco/static_overbroad_top100_l9_l16/static_overbroad_top100.svg`

Caption stance:

```text
Static hard suppression triggers frequently on grounded object head-steps and produces grounded-object reduction/disappearance.
```
