# Section III-D/F Analysis Notes

This note consolidates the analysis for:

- **III-D. Hallucination Heads as Text-Side Actuators**
- **III-F. Static Suppression Is an Over-Broad Intervention**

The purpose is to support the analysis section before introducing the method. The central claim is not that the selected heads are hallucination detectors. The safer and better-supported interpretation is:

> Selected heads are text-side actuator channels. They provide intervention leverage over object generation by routing attention through post-image textual context. However, because the same heads are also active during grounded object generation, static hard suppression is an over-broad intervention.

---

## Data Sources

### D. Actuator Characterization

Main files:

- `LLaVA/results/coco/selected_head_actuator_analysis_l9_l16_k100/head_actuator_group_summary.csv`
- `LLaVA/results/coco/selected_head_actuator_analysis_l9_l16_k100/head_actuator_direction_counts.csv`
- `LLaVA/results/coco/selected_head_actuator_analysis_l9_l16_k100/head_actuator_analysis_summary.json`
- `LLaVA/results/coco/selected_head_actuator_analysis_l9_l16_k100/selected_vs_nonselected_actuator_metrics.svg`

Comparison:

- selected: top-100 heads from the L9-L16 actuator-ranked pool
- non-selected: remaining 156 heads in the same L9-L16 window

This is a head-pool characterization, not a token-level detector test.

### F. Static Suppression Exposure and Collateral Damage

Main files:

- `LLaVA/results/coco/static_trigger_rate_bar_tau0p4/static_trigger_rate_summary.csv`
- `LLaVA/results/coco/static_trigger_rate_bar_tau0p4/static_trigger_and_grounded_oversuppression_summary.csv`
- `LLaVA/results/coco/static_trigger_rate_bar_tau0p4/static_trigger_and_grounded_oversuppression_refined.svg`
- `LLaVA/results/coco/adhh_hard_tau0p4_removal_loss_disappear/adhh_removal_loss_summary.json`
- `LLaVA/results/coco/adhh_hard_tau0p4_removal_loss_disappear/adhh_removal_loss_rows.csv`

Important caveat:

- The trigger-rate figure currently uses the available top-150 L9-L16 trace from `method_figure_source_trace_n100_k150_l9_16`.
- Recomputing the same trigger diagnostic on the top-100 L9-L16 subset gives the same qualitative conclusion: grounded object steps are still frequently suppressed.

---

# III-D. Hallucination Heads as Text-Side Actuators

## Question

If the selected heads are important for hallucination mitigation, what role do they play?

The detector-centric view would say:

> These heads detect that the next object token is hallucinated.

The actuator view says:

> These heads are channels through which post-image textual context influences object generation. They are useful intervention points, but they do not need to classify the next token as hallucinated.

The analysis supports the actuator view.

## Selected vs Non-Selected Head Metrics

| metric | selected top-100 | non-selected | ratio |
|---|---:|---:|---:|
| text-side mass `Itext_all` | 0.408 | 0.254 | 1.61x |
| contrastive TOI score `C_toi H-G` | 6.604 | 0.716 | 9.22x |
| log-TOI gap H-G | 0.259 | 0.092 | 2.82x |
| image mass drop G-H | 0.0224 | 0.00724 | 3.09x |

Interpretation by metric:

- `Itext_all`: selected heads attend more strongly to post-image text-side context. This supports the claim that they have stronger language-context leverage.
- `C_toi H-G`: selected heads show a much larger hallucinated-minus-grounded text-over-image shift. This means they are not merely text-heavy; their text-over-image reliance changes more strongly in hallucination states.
- `log-TOI gap H-G`: the same trend remains after using a more stable log-ratio view, so the contrastive signal is not only a raw-ratio outlier artifact.
- `image mass drop G-H`: selected heads show a stronger reduction in image-token mass when moving from grounded to hallucinated object steps. This links text-side actuation to weakened visual routing.

## Directionality Counts

Among selected top-100 heads:

| direction | count | rate |
|---|---:|---:|
| positive text-mass gap H-G | 64/100 | 64% |
| positive image drop G-H | 76/100 | 76% |
| positive raw TOI gap H-G | 82/100 | 82% |
| positive log TOI gap H-G | 86/100 | 86% |

Among non-selected heads:

| direction | count | rate |
|---|---:|---:|
| positive text-mass gap H-G | 74/156 | 47.4% |
| positive image drop G-H | 90/156 | 57.7% |
| positive raw TOI gap H-G | 68/156 | 43.6% |
| positive log TOI gap H-G | 103/156 | 66.0% |

This shows that selected heads are directionally more aligned with the actuator interpretation.

## Complementarity of the Two Selection Axes

The text-leverage percentile and contrastive percentile are weakly negatively correlated across the L9-L16 head pool:

```text
Pearson r = -0.138
```

This matters because the two features are not redundant:

- text-side mass identifies **where intervention can move object generation**
- contrastive TOI identifies **where hallucinated object generation becomes more text-over-image biased**

The fused selection is therefore better interpreted as:

```text
leverage axis + hallucination-specific bias axis
```

not as simply selecting the highest text-attention heads.

## Correct Interpretation

The selected heads should be described as **text-side actuators**:

> They carry stronger post-image textual context, show larger hallucinated-minus-grounded text-over-image bias, and lose more image-token mass during hallucinated object generation.

They should not be described as **hallucination detectors**:

> High text-side reliance is also common during grounded object generation, so selected-head activation alone does not reliably distinguish hallucinated from grounded object tokens.

## Paper-Ready Paragraph

The selected heads are better understood as text-side actuators rather than hallucination detectors. Compared with non-selected heads in the same L9-L16 window, selected heads carry substantially larger post-image text-side mass (0.408 vs. 0.254) and exhibit a much stronger hallucinated-minus-grounded text-over-image contrast (6.604 vs. 0.716). This shift remains visible in log-ratio form (0.259 vs. 0.092), and selected heads also show a larger image-token mass drop when moving from grounded to hallucinated object steps (0.022 vs. 0.007). These results indicate that the selected heads are intervention-relevant channels through which language-context support can dominate object generation while visual-token routing weakens. Crucially, this does not imply that the heads detect hallucination. Their role is actuator-like: they provide a control point for reducing text-prior support, but the decision of when to intervene cannot be recovered from head identity alone.

## What Not To Claim

Avoid:

```text
Selected heads detect hallucination.
Text-side mass identifies hallucinated object tokens.
The selected head pool is a hallucination classifier.
```

Use:

```text
Selected heads are text-side actuator channels.
Text-side mass is an intervention leverage signal.
Contrastive TOI is a hallucination-specific bias signal.
The fused head pool combines leverage and specificity.
```

---

# III-F. Static Suppression Is an Over-Broad Intervention

## Question

If selected heads are useful actuator channels, why not simply suppress them statically?

The analysis here should not criticize AD-HH as ineffective. Static suppression does reduce hallucination. The point is more precise:

> Static hard suppression is effective but coarse. It suppresses selected actuator heads in grounded object states almost as often as in hallucinated object states, and this broad exposure produces measurable grounded-object loss.

## Static Trigger Exposure

The static hard gate triggers when:

```text
text_mass >= tau
```

with:

```text
tau = 0.4
```

Using the available top-150 L9-L16 trace:

| label | triggered head-steps | total head-steps | trigger rate |
|---|---:|---:|---:|
| grounded object | 47,934 | 107,400 | 44.63% |
| hallucinated object | 8,321 | 18,150 | 45.85% |

Difference:

```text
45.85% - 44.63% = 1.22 percentage points
```

This means the static gate is not selective at the object-state level. Grounded and hallucinated object steps cross the hard text-mass threshold at nearly the same rate.

### Top-100 Check

Because the current main team setting often uses top-100 heads, the same trigger diagnostic was recomputed by restricting the top-150 trace to the top-100 L9-L16 heads.

| label | triggered head-steps | total head-steps | trigger rate |
|---|---:|---:|---:|
| grounded object | 28,089 | 71,600 | 39.23% |
| hallucinated object | 5,222 | 12,100 | 43.16% |

Difference:

```text
43.16% - 39.23% = 3.93 percentage points
```

This is somewhat more separated than the top-150 trace, but it still supports the same conclusion: grounded object steps are frequently exposed to static suppression. A grounded-object trigger rate near 40% is too high to interpret the hard gate as hallucination-specific.

## Behavioral Grounded-Object Outcome

Comparing greedy captions with hard-suppressed captions:

| metric | count | rate |
|---|---:|---:|
| grounded object nodes | 1,299 | - |
| grounded object nodes reduced | 363/1,299 | 27.94% |
| grounded object nodes partially reduced | 175/1,299 | 13.47% |
| grounded object nodes disappeared | 188/1,299 | 14.47% |
| grounded mentions removed | 490/3,290 | 14.89% |

Here, a **grounded object node** means a unique grounded COCO object category in an image-caption pair. This is cleaner than mention-level counts because repeated mentions of the same object do not dominate the statistic.

The key behavioral result:

```text
27.9% of grounded object nodes are reduced.
14.5% disappear entirely.
```

This is the collateral-damage evidence for static hard suppression.

## Hallucination Reduction Still Happens

The same hard suppression also reduces hallucination:

| metric | count | rate |
|---|---:|---:|
| hallucinated mentions removed | 372/548 | 67.88% |
| images with hallucination removed | 220/500 | 44.0% |
| CHAIRs greedy -> hard | 0.534 -> 0.342 | lower is better |
| CHAIRi greedy -> hard | 0.1428 -> 0.0886 | lower is better |

This should be stated clearly:

> Static suppression is not useless. It works, but its intervention unit is broad.

## Representative Mixed-Outcome Case

Image:

```text
COCO_val2014_000000208748.jpg
image id: 208748
```

This image contains all three grounded outcomes within the same sample:

| outcome | object transition |
|---|---|
| no reduction | dining table 5 -> 5 |
| no reduction | sandwich 1 -> 1 |
| no reduction / increased mention | wine glass 1 -> 3 |
| partial reduction | person 2 -> 1 |
| disappeared | cup 1 -> 0 |
| disappeared | fork 1 -> 0 |
| disappeared | knife 1 -> 0 |

The same example also removes hallucinated objects:

```text
base hallucinated: bowl, spoon, bottle
hard suppressed:  none
```

Interpretation:

> Static suppression can remove hallucinated content, but within the same image it can also partially reduce or completely remove grounded content.

This is a good qualitative example for the over-suppression argument because it shows intended and unintended effects together.

## Refined Argument For Section III-F

The argument should be structured as a two-stage funnel:

```text
1. Static gate exposure:
   selected heads cross the hard threshold for grounded objects almost as often as for hallucinated objects.

2. Behavioral outcome:
   this broad exposure materializes as grounded-object reduction and disappearance.
```

Do not use the caption-disappearance analysis alone. The stronger argument is:

```text
hard trigger rate is broad -> grounded object nodes are actually reduced/disappeared
```

This connects the mechanism to the caption-level outcome.

## Paper-Ready Paragraph

Static hard suppression exposes the limitation of treating selected heads as hallucination detectors. Under the hard rule `text_mass >= 0.4`, suppression is triggered for grounded and hallucinated object steps at nearly the same rate: 44.63% of grounded selected head-steps and 45.85% of hallucinated selected head-steps in the available top-150 L9-L16 trace. Even when restricted to the top-100 heads, grounded object steps remain heavily exposed, with a 39.23% trigger rate compared with 43.16% for hallucinated object steps. This broad exposure produces measurable collateral damage. Comparing greedy captions with hard-suppressed captions, 27.94% of grounded object nodes are reduced, and 14.47% disappear entirely. Thus, static suppression is effective but coarse: it can remove hallucinated objects, but because the same text-side actuator heads also support grounded object realization, a fixed hard gate can suppress normal grounded content as well.

## What Not To Claim

Avoid:

```text
Static suppression fails.
AD-HH is wrong.
The hard gate never distinguishes hallucination.
```

Use:

```text
Static suppression is effective but over-broad.
The hard gate exposes grounded object generation to suppression.
Selected heads are useful actuator channels, but head identity plus a fixed threshold is not enough for token-state selectivity.
```

---

# How D and F Connect

Section D establishes:

```text
Selected heads are actuator channels.
```

Section F establishes:

```text
Static suppression of actuator channels is too broad.
```

Together:

```text
The analysis motivates dynamic, token-state-dependent suppression without requiring the selected heads to be hallucination detectors.
```

This is the clean conceptual bridge into the method section.

