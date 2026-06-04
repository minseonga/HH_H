# Section III-D/F 분석 정리

이 문서는 현재 로컬에 있는 결과 파일만 기준으로 Section III-D와 III-F에 들어갈 수 있는 claim, 수치, 해석을 정리한다. 핵심 전제는 다음과 같다.

- 온라인 ratio 정의는 팀원 method 코드와 동일하게 고정한다.

  \[
  r_{t,l,h}=\frac{T_{t,l,h}}{T_{t,l,h}+I_{t,l,h}+\epsilon}
  \]

- 여기서 \(T\)는 post-image text-side attention mass, \(I\)는 image-token attention mass이다. system prefix는 ratio 분모에 넣지 않는다.
- DEACT의 caption-level collateral, 즉 exact `dynamic_l9_l16_k100_s1.0_q8_tau0.90`가 grounded object를 얼마나 보존하는지는 아직 최종 측정되지 않았다. 이 비교는 Section V로 넘긴다.
- 따라서 III-F에서는 "dynamic이 static보다 collateral을 줄인다"를 주장하지 않는다. III-F의 역할은 static/hard suppression이 over-broad intervention이라는 문제를 분석하는 것이다.

## Source Files

현재 문서에 사용한 주요 source는 다음과 같다.

| 용도 | 파일 |
|---|---|
| ratio detector / trigger / delta diagnostic | `LLaVA/results/coco/text_image_ratio_diagnostics_top100_l9_l16/text_image_ratio_detector_summary.json` |
| selected vs non-selected actuator 지표 | `LLaVA/results/coco/selected_head_actuator_analysis_l9_l16_k100/head_actuator_group_summary.csv` |
| static hard grounded outcome | `LLaVA/results/coco/adhh_hard_tau0p4_removal_loss_disappear/adhh_removal_loss_summary.json` |
| static logprob perturbation diagnostic | `LLaVA/results/coco/adhh_static_hall_ground_touch_figure/adhh_static_hall_ground_touch_summary.json` |
| static over-broad auxiliary summary | `LLaVA/results/coco/static_overbroad_top100_l9_l16/static_overbroad_top100_summary.json` |

주의: `text_image_ratio_diagnostics_top100_l9_l16`는 available n=100 trace에서 exact top-100 head file을 필터링한 diagnostic이다. 해당 trace에는 exact top-100 중 90개 head만 포함되어 있다. 따라서 이 숫자는 method framing 검증용이며, 최종 논문 수치로 고정하려면 exact top-100 trace를 다시 저장해야 한다.

---

# D. Hallucination Heads as Text-Side Actuators

## D의 중심 claim

선택된 hallucination-related heads는 hallucination detector라기보다 text-side actuator로 해석하는 것이 더 정직하다. 즉 이 head들은 "다음 object token이 hallucinated인지"를 sharp하게 판별하는 detector가 아니라, decoding 중 text-side context가 object generation에 영향을 주는 통로이다.

이 해석은 세 단계로 정리된다.

1. C절에서 \(r=T/(T+I)\)는 token-level detector로 약하다.
2. 그러나 selected heads는 hallucinated object step에서 image-side routing이 더 약해지는 관찰적 패턴을 보인다.
3. suppression diagnostic에서는 hallucinated object가 grounded object보다 훨씬 더 크게 log-probability perturbation을 받는다.

따라서 selected heads는 "hallucination-only detector"가 아니라 "text-side influence를 줄이면 object generation이 흔들리는 intervention point"로 보는 것이 맞다.

## D-1. Detector가 아니라는 연결: C절 결과

Token-level detector test에서 \(r=T/(T+I)\)의 H-vs-G AUC는 다음과 같다.

| 신호 | 단위 | AUC |
|---|---:|---:|
| \(r=T/(T+I)\) | object token-level | 0.594 |

Token-level ratio 분포:

| label | n | mean | q25 | median | q75 | q90 |
|---|---:|---:|---:|---:|---:|---:|
| grounded | 716 | 0.818 | 0.769 | 0.841 | 0.895 | 0.926 |
| hallucinated | 121 | 0.853 | 0.815 | 0.869 | 0.907 | 0.935 |

해석:

- hallucinated object가 평균적으로 더 text-heavy한 것은 맞다.
- 하지만 grounded object도 높은 text-side reliance를 보인다.
- 따라서 \(r\)만으로 hallucinated object token을 detector처럼 분리하기 어렵다.

논문에 넣을 문장:

> The ratio \(T/(T+I)\) is higher on hallucinated object steps on average, but it does not form a reliable token-level detector. Its token-level AUC is only 0.594, and grounded object steps also occupy the high text-reliance regime. This motivates interpreting the selected heads not as detectors, but as text-side actuation channels.

## D-2. Selected heads의 관찰적 actuator 특성

L9-L16, top-100 selected heads와 non-selected heads 비교:

| group | n_heads | \(I_{text,all}\) | \(C_{toi,H-G}\) | LogTOI H-G | Image drop G-H |
|---|---:|---:|---:|---:|---:|
| selected | 100 | 0.348 | 6.311 | 0.299 | 0.02699 |
| non-selected | 156 | 0.291 | 0.0657 | -0.235 | 0.00663 |

Image drop의 세부값:

| group | Img mass on H | Img mass on G | G-H image drop |
|---|---:|---:|---:|
| selected | 0.07476 | 0.10175 | 0.02699 |
| non-selected | 0.08051 | 0.08714 | 0.00663 |

Image drop ratio:

\[
\frac{0.02699}{0.00663}=4.07
\]

해석:

- \(I_{text,all}\)과 \(C_{toi,H-G}\)는 head selection score의 구성 축과 직접 연결되어 있으므로, 그것만으로 actuator claim을 증명하면 순환 논리가 된다.
- 하지만 `Image drop G-H`는 selection score 그 자체가 아니다. selected heads는 hallucinated object step에서 grounded object step보다 image-token mass가 더 크게 빠진다.
- 즉 selected heads에서는 hallucinated object generation 시 text-over-image shift뿐 아니라 visual routing weakening이 함께 나타난다.

논문에서 쓸 수 있는 안전한 표현:

> By construction, the selected pool is enriched for text-side mass and text-over-image contrast. More importantly, an additional non-selection statistic shows that these heads also exhibit a stronger visual-routing drop: their image-token mass decreases by 0.02699 from grounded to hallucinated object steps, compared with 0.00663 for non-selected heads. This 4.07x larger image-mass drop suggests that hallucinated object generation is accompanied not only by increased text-side reliance, but also by weakened visual-token routing in the selected heads.

피해야 할 표현:

- "selected heads detect hallucination"
- "high text mass is hallucination evidence"
- "C_toi가 높으므로 actuator다"

대신 쓸 표현:

- text mass = leverage signal
- contrastive TOI = group-level hallucination-specific bias
- image mass drop = non-circular visual weakening evidence
- actuator = causal diagnostic까지 포함했을 때 정당화되는 해석

## D-3. Causal fragility diagnostic

Static object log-probability perturbation diagnostic:

\[
\Delta \log p(y_t)=\log p_{\text{base}}(y_t)-\log p_{\text{suppressed}}(y_t)
\]

| label | n | mean \(\Delta \log p\) | median \(\Delta \log p\) | q75 | q90 | positive drop fraction | top-1 changed fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| grounded object | 200 | 0.0809 | 0.0196 | 0.1959 | 0.4471 | 0.625 | 0.135 |
| hallucinated object | 200 | 0.3864 | 0.2208 | 0.6259 | 1.0962 | 0.805 | 0.345 |

Mean logprob drop ratio:

\[
\frac{0.3864}{0.0809}=4.77
\]

Top-1 changed ratio:

\[
\frac{0.345}{0.135}=2.56
\]

해석:

- selected/text-side suppression은 hallucinated object의 target likelihood를 grounded object보다 훨씬 크게 흔든다.
- grounded object도 영향을 받지만, hallucinated object가 더 fragile하다.
- 이것이 actuator 해석의 인과적 핵심이다. "head가 hallucination을 detect한다"가 아니라, "head를 누르면 hallucinated object likelihood가 더 쉽게 붕괴한다"가 맞는 표현이다.

단서:

- 이 diagnostic은 현재 exact `dynamic_l9_l16_k100_s1.0_q8_tau0.90` caption run에서 직접 나온 것이 아니다.
- D절에 causal evidence로 쓰려면, 최종적으로 exact top-100 L9-L16 설정으로 재측정하는 것이 안전하다.

논문에 넣을 문장:

> The actuator interpretation is supported by an intervention diagnostic. When the text-side contribution of selected heads is suppressed at object steps, hallucinated objects show a mean target log-probability drop of 0.3864, whereas grounded objects drop by only 0.0809. This 4.77x fragility gap indicates that the selected heads are not reliable detectors of hallucination, but they are effective control points: attenuating their text-side pathway disproportionately destabilizes unsupported object tokens.

---

# F. Static Suppression Is an Over-Broad Intervention

## F의 중심 claim

Static hard suppression의 문제는 "어떤 head를 고르는가"만이 아니다. 더 근본적인 문제는 head identity만으로 intervention을 걸면 grounded object step에도 넓게 작동한다는 점이다.

즉 static suppression은 hallucination-prone moment를 token-level로 판별하지 못한다. selected head가 grounded object generation에도 사용되기 때문에, hard suppression은 hallucinated object를 줄이는 동시에 grounded object realization도 손상시킬 수 있다.

## F-1. Ratio trigger는 grounded에도 많이 걸린다

현재 \(r=T/(T+I)\), \(\tau=0.9\) 기준 diagnostic:

Token-level trigger:

| label | n tokens | triggered | trigger rate |
|---|---:|---:|---:|
| grounded | 716 | 164 | 22.9% |
| hallucinated | 121 | 37 | 30.6% |

Head-step trigger:

| label | n head-steps | triggered | trigger rate |
|---|---:|---:|---:|
| grounded | 64,440 | 32,647 | 50.7% |
| hallucinated | 10,890 | 5,989 | 55.0% |

해석:

- Token-level에서는 hallucinated trigger rate가 grounded보다 높지만, 차이가 detector로 쓰기에 충분하지 않다.
- 실제 intervention 단위인 head-step에서는 grounded도 50.7%가 \(\tau=0.9\) 이상이다.
- 따라서 "selected heads + high text/image ratio"는 hallucination-only trigger가 아니다.

논문에 넣을 문장:

> At the intervention unit, the over-breadth becomes more apparent. Using the same \(r=T/(T+I)\) signal and \(\tau=0.9\), 55.0% of hallucinated object head-steps cross the threshold, but so do 50.7% of grounded object head-steps. Thus, a hard threshold over selected heads would expose grounded object generation to nearly the same kind of text-side attenuation as hallucinated generation.

주의:

- 이 결과는 exact top-100 중 90개 head만 포함된 trace에서 계산한 diagnostic이다.
- 최종 paper number는 exact top-100 trace로 다시 계산해야 한다.

## F-2. Dynamic gate도 selective detector가 아니다

팀원 method의 suppression strength는 다음과 같다.

\[
\delta_{t,l,h}
=
\mathrm{clip}
\left(
s \cdot S(l,h)\exp(q(r_{t,l,h}-\tau)),
0,1
\right)
\]

현재 diagnostic 설정:

| hyperparameter | value |
|---|---:|
| \(s\) | 1.0 |
| \(q\) | 8 |
| \(\tau\) | 0.9 |
| layers | L9-L16 |
| top-k | 100 |

Diagnostic \(\delta\) distribution:

Token-level mean \(\delta\):

| label | mean | q25 | median | q75 | q90 |
|---|---:|---:|---:|---:|---:|
| grounded | 0.625 | 0.528 | 0.644 | 0.762 | 0.845 |
| hallucinated | 0.672 | 0.577 | 0.698 | 0.781 | 0.855 |

Head-step \(\delta\):

| label | mean | q25 | median | q75 | q90 |
|---|---:|---:|---:|---:|---:|
| grounded | 0.625 | 0.263 | 0.725 | 1.000 | 1.000 |
| hallucinated | 0.672 | 0.372 | 0.792 | 1.000 | 1.000 |

해석:

- \(\delta\)는 hallucinated object에서 약간 더 높지만, grounded에도 상당히 크게 걸린다.
- 따라서 method를 "critical threshold를 넘는 hallucination-prone step에만 selective하게 suppress한다"고 설명하면 안 된다.
- \(\tau=0.9\)는 hard cutoff가 아니다. \(r<\tau\)여도 exponential gate는 0이 아니다.

예:

\[
\exp(8(0.841-0.9)) \approx 0.624
\]

즉 grounded median token-level \(r=0.841\)에서도 score가 높으면 \(\delta\approx0.62\)가 가능하다.

논문에서 피해야 할 표현:

- "suppresses only after crossing a critical threshold"
- "selectively intervenes only on hallucination-prone steps"
- "dynamic gate detects hallucinated object generation"

대신 쓸 표현:

- "score-weighted continuous attenuation"
- "broad but graded text-side attenuation"
- "the gate controls magnitude, not binary detection"

논문에 넣을 문장:

> The dynamic gate should not be interpreted as a selective hallucination detector. Because the exponential term is nonzero below \(\tau\), grounded object steps can still receive substantial attenuation. In the available diagnostic trace, token-level mean \(\delta\) is 0.625 for grounded objects and 0.672 for hallucinated objects. The method is therefore better described as broad but graded text-side attenuation, rather than thresholded hallucination-only suppression.

## F-3. Static hard suppression의 grounded collateral

Static hard tau=0.4 comparison, COCO n=500:

Overall metrics:

| method | CHAIRs | CHAIRi | Bleu1 | avg caption length |
|---|---:|---:|---:|---:|
| greedy | 0.534 | 0.1428 | 0.1809 | 87.93 |
| AD-HH hard tau=0.4 | 0.342 | 0.0886 | 0.1756 | 89.80 |

Hallucinated object outcome:

| quantity | count/rate |
|---|---:|
| base hallucinated mentions | 548 |
| removed hallucinated mentions | 372 |
| hallucinated removal rate | 67.9% |
| target hallucinated mentions | 401 |
| added hallucinated mentions | 225 |

Grounded object outcome:

| quantity | count/rate |
|---|---:|
| base grounded mentions | 3290 |
| removed grounded mentions | 490 |
| grounded mention loss rate | 14.9% |
| base grounded object nodes | 1299 |
| reduced grounded nodes | 363 |
| grounded node reduction rate | 27.9% |
| partially reduced grounded nodes | 175 |
| partial reduction rate | 13.5% |
| disappeared grounded nodes | 188 |
| disappearance rate | 14.5% |
| images with any grounded loss | 248 / 500 |
| image-level grounded loss rate | 49.6% |

해석:

- Static hard suppression은 hallucinated mentions를 많이 제거한다. 제거율은 67.9%이다.
- 동시에 grounded mentions도 490개 제거되며, base grounded mentions 대비 14.9% 손실이다.
- grounded object node 기준으로는 27.9%가 감소했고, 그중 14.5%는 완전히 사라졌다.
- removed grounded mentions 490개 중 disappeared-grounded mention은 245개로, grounded loss의 50.0%가 완전 소멸에서 온다.

논문에 넣을 문장:

> Static hard suppression is effective at reducing hallucinated mentions, removing 372 of 548 baseline hallucinated mentions (67.9%). However, it also removes 490 grounded mentions out of 3290 (14.9%). At the object-node level, 363 of 1299 grounded nodes are reduced (27.9%), and 188 nodes disappear entirely (14.5%). This shows that the intervention is not hallucination-specific: suppressing selected heads can also erase grounded object realization.

## F-4. F절의 정확한 결론

F절에서 말할 수 있는 것:

1. \(r=T/(T+I)\)는 method와 정합되는 신호이다.
2. 같은 ratio 신호를 보더라도 grounded object head-step이 매우 자주 high-ratio regime에 들어온다.
3. static hard suppression은 hallucinated object를 줄이지만 grounded object도 실제로 감소시킨다.
4. dynamic gate도 token detector가 아니다. 다만 hard binary intervention이 아니라 continuous attenuation이라는 점이 다르다.

F절에서 아직 말하면 안 되는 것:

1. "DEACT가 static보다 grounded collateral을 줄인다."
2. "DEACT는 hallucination-prone token만 selective하게 suppress한다."
3. "tau=0.9가 grounded와 hallucinated를 clean하게 나눈다."

이유:

- exact `dynamic_l9_l16_k100_s1.0_q8_tau0.90`의 caption-level grounded collateral이 아직 측정되지 않았다.
- 현재 \(\delta\) diagnostic은 dynamic gate가 grounded에도 강하게 작동할 수 있음을 보여준다.
- 따라서 DEACT의 behavioral advantage는 Section V에서 exact run으로 별도 증명해야 한다.

## F절 마지막 문장 초안

> These results identify the limitation of static head suppression. The selected heads provide useful intervention leverage, but head identity and high text-image ratio do not uniquely identify hallucinated object generation. Static hard suppression therefore acts as an over-broad intervention: it removes hallucinated objects, but also reduces grounded object realization. This motivates replacing binary suppression with score-weighted continuous attenuation, while leaving the behavioral question of grounded-object preservation to the evaluation section.

---

# 최종 Paper Framing

## C-D-F 연결

논리 흐름은 다음과 같이 정리하는 것이 가장 안전하다.

1. C: \(r=T/(T+I)\)는 hallucination detector로 약하다. Token-level AUC는 0.594이다.
2. D: 그래도 selected heads는 text-side actuator로 볼 수 있다. hallucinated object step에서 selected heads의 image mass drop이 non-selected보다 4.07배 크고, suppression diagnostic에서 hallucinated object logprob drop이 grounded보다 4.77배 크다.
3. F: static hard suppression은 이 actuator를 너무 거칠게 사용한다. Grounded object head-step도 50.7%가 \(\tau=0.9\) 이상이고, 실제 static hard run에서 grounded object node 27.9%가 감소한다.
4. Method: 따라서 method는 detector-based selective gating이 아니라 score-weighted continuous attenuation으로 써야 한다.
5. V: exact DEACT가 static보다 grounded collateral을 줄이는지는 caption-level evaluation에서 증명해야 한다.

## Method 표현 수정

기존에 쓰면 위험한 표현:

> DEACT selectively suppresses hallucination-prone steps when the text-image ratio crosses a critical threshold.

수정된 표현:

> DEACT applies score-weighted continuous attenuation to selected text-side actuator heads. The online text-image ratio does not act as a reliable hallucination detector; instead, it modulates the magnitude of attenuation. This avoids treating head selection as a binary hallucination decision and turns hard text suppression into graded control.

## 현재 남은 필수 실험

최우선:

```bash
cd ~/Hallucination-Attribution
GPU_ID=6 \
NUM_SAMPLES=100 \
SEED=42 \
bash LLaVA/bash_scripts/soft_routing/run_exact_deact_grounded_collateral.sh
```

확인할 파일:

```text
LLaVA/results/coco/exact_deact_l9_l16_k100_s1.0_q8.0_tau0.90_n100_seed42/
  grounded_collateral_dynamic/adhh_removal_loss_summary.json
```

확인할 수치:

```text
grounded_reduced_node_rate
grounded_disappeared_node_rate
hallucinated_removal_rate
```

판정:

- exact DEACT의 `grounded_disappeared_node_rate`가 static hard의 14.5%보다 명확히 낮으면, continuous attenuation 서사가 살아난다.
- exact DEACT의 grounded collateral이 static과 비슷하면, `s=1.0, q=8, tau=0.9`는 너무 aggressive한 설정이다.
