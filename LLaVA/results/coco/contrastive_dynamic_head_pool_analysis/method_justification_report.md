# Contrastive Dynamic Head Pool Justification

- ranked head file: `../ADHH/LLaVA/results/coco/llava-v1.5-7b_base_original_qa_n3000/surrogate_hh_scores/surrogate_score_zoo/ranked_heads_global__itext_all__C_toi_HminusG.json`
- score: `global__itext_all__C_toi_HminusG`
- description: global selection over rank-percentile combo: 0.5*P(itext_all) + 0.5*P(C_toi_HminusG). local selection orders layers by first global occurrence, then round-robins layer-local ranks.

## 핵심 해석

이 head pool은 AD-HH head를 그대로 가져온 것이 아니라, 두 축을 rank-percentile로 결합한 pool이다.
첫 축은 intervention 대상인 image 이후 text-side attention mass이고, 둘째 축은 hallucinated object step에서 grounded object step보다 text-over-image ratio가 커지는 contrast이다.
따라서 이 pool의 의미는 `text leverage`와 `hallucination-specific text-over-image reliance`의 교집합이다.

## Pool 구조

- top20 heads: `L31H27 L14H16 L31H13 L14H20 L29H10 L28H25 L26H9 L31H0 L24H23 L23H18 L15H25 L26H28 L16H2 L29H15 L27H12 L18H9 L18H23 L26H14 L16H17 L30H10`
- top100 layer distribution: L13:5, L14:12, L15:13, L16:10, L17:3, L18:4, L19:6, L20:6, L21:2, L22:1, L23:4, L24:5, L25:1, L26:5, L27:4, L28:3, L29:2, L30:7, L31:7
- top100 AD-HH overlap: 14/20 recall=0.7000, selected fraction=0.1400
- top100 mean Itext_all=0.4230, rank>200 mean Itext_all=0.1327
- top100 mean RawTOI H-G gap=64.9125, rank>200 gap=-2.3209
- top100 mean image drop(nonhall-hall)=0.0378, rank>200 image drop=0.0106

## Surrogate Consistency

- `layerprior_top5mean_alpha0p5__itext_all__C_toi_HminusG`: Spearman with target rank percentile=0.9956
- `global__txt_attn_raw_all__C_toi_HminusG`: Spearman with target rank percentile=0.9950
- `layerprior_top5mean_alpha0p5__txt_attn_raw_all__C_toi_HminusG`: Spearman with target rank percentile=0.9914
- `layerprior_top5mean_alpha1p0__itext_all__C_toi_HminusG`: Spearman with target rank percentile=0.9848
- `layerprior_top5mean_alpha1p0__txt_attn_raw_all__C_toi_HminusG`: Spearman with target rank percentile=0.9812
- `layerprior_top5mean_alpha2p0__itext_all__C_toi_HminusG`: Spearman with target rank percentile=0.9516
- `layerprior_top5mean_alpha2p0__txt_attn_raw_all__C_toi_HminusG`: Spearman with target rank percentile=0.9492
- `global__itext_all__C_logtoi_HminusG`: Spearman with target rank percentile=0.9144

## Empirical Link

- greedy n=500: CHAIRs=0.5460, CHAIRi=0.1476, F1=0.7638
- AD-HH k20 n=500: CHAIRs=0.3660, CHAIRi=0.0964, F1=0.7779
- best local dynamic run: `dynamic_k100_s0.7_q10.0_tau0.90` CHAIRs=0.3300, CHAIRi=0.1026, BLEU1=0.1789, F1=0.7588

## 논문/보고서에 넣을 수 있는 주장

1. AD-HH는 fixed small actuator set이지만, teammate method는 larger hallucination-relevant text-leverage pool을 사용한다.
2. Head selection 자체가 suppression target과 정렬되어 있다. generated-prefix raw text attention이 아니라 image 이후 text-side slice를 쓰므로 실제 intervention slice와 같은 축이다.
3. Contrastive term 때문에 단순 text-heavy head가 아니라 hallucinated object step에서 text-over-image reliance가 더 커지는 head를 우선한다.
4. Online gate는 head가 매 step 실제로 text-dominant일 때만 강하게 suppress한다. 즉 offline head pool은 `where`, online ratio는 `when/strength` 역할을 한다.
5. 우리 AD-HH 분석의 결론과 맞는다. text_mass는 hallucination evidence가 아니라 actuator leverage이고, dynamic method는 이 leverage를 hallucination contrast 및 online state와 결합한다.

## 남는 약점

- Recall 하락은 여전히 있다. 큰 top-k pool에서 grounded object도 영향을 받기 때문이다.
- 그래서 현재 추가 실험의 목적은 CHAIR를 유지하거나 더 낮추면서 concentration gate로 suppression을 더 선별적으로 만들어 recall/F1 손실을 줄이는 것이다.
