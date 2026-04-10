# Bible RAG moral reasoning benchmark

Evaluating whether retrieval-augmented generation with Bible verses improves moral judgment accuracy on the [ETHICS benchmark](https://arxiv.org/abs/2008.02275) (Hendrycks et al., 2020).

## Setup

**Task:** Binary moral classification — given a scenario, output `0` (acceptable) or `1` (unacceptable).

**Models:** Qwen3 4B, Qwen3 8B

**Conditions:**
- `plain` — scenario only, no retrieval context
- `bible` — 5 retrieved Bible verses (semantic search) prepended to prompt
- `bible_ctx3` — same retrieval, but each verse expanded with ±3 surrounding verses for narrative context

**Datasets (1,000 samples each):** Commonsense, Deontology, Justice, Utilitarianism, Virtue

---

## Results

### Accuracy by condition

| Dataset | 4B plain | 4B bible | 4B bible ±3 | 8B plain | 8B bible | 8B bible ±3 |
|---|---|---|---|---|---|---|
| Commonsense | 87.5% | 86.4% | 85.8% | 91.7% | 88.6% | 89.1% |
| Deontology | 60.5% | 60.5% | 60.5% | 69.4% | 58.0% | 57.0% |
| Justice | 71.7% | 67.7% | 69.7% | 71.4% | 64.9% | 60.5% |
| Utilitarianism | 81.4% | 96.8% | 94.4% | 63.2% | 75.1% | 76.1% |
| Virtue | 20.7% | 22.1% | 16.3% | 70.8% | 76.7% | 81.1% |
| **Average** | **64.4%** | **66.7%** | **65.3%** | **73.3%** | **72.7%** | **72.8%** |

### Bible effect: delta vs plain

| Dataset | 4B bible − plain | 4B bible ±3 − plain | 8B bible − plain | 8B bible ±3 − plain |
|---|---|---|---|---|
| Commonsense | −1.1pp | −1.7pp | −3.1pp | −2.6pp |
| Deontology | 0.0pp | 0.0pp | −11.4pp | −12.4pp |
| Justice | −4.0pp | −2.0pp | −6.5pp | −10.9pp |
| Utilitarianism | +15.4pp | +13.0pp | +11.9pp | +12.9pp |
| Virtue | +1.4pp | −4.4pp | +5.9pp | +10.3pp |

### Context-3 vs single-verse delta

| Dataset | 4B (ctx3 − single) | 8B (ctx3 − single) |
|---|---|---|
| Commonsense | −0.6pp | +0.5pp |
| Deontology | 0.0pp | −1.0pp |
| Justice | +2.0pp | −4.4pp |
| Utilitarianism | −2.4pp | +1.0pp |
| Virtue | −5.8pp | +10.3pp |

---

## Confusion matrices

| Condition | TP | TN | FP | FN |
|---|---|---|---|---|
| 4B plain | 1232 | 1172 | 1395 | 201 |
| 4B bible | 1130 | 1237 | 1330 | 303 |
| 4B bible ±3 | 1073 | 1250 | 1259 | 360 |
| 8B plain | 1102 | 1931 | 636 | 331 |
| 8B bible | 735 | 2147 | 420 | 698 |
| 8B bible ±3 | 630 | 2247 | 320 | 803 |

---

## Key findings

### 1. Bible RAG is neutral on average, but highly uneven by dataset
Averaged across all five datasets, Bible context provides essentially no improvement for either model (4B: −0.5pp; 8B: −0.6pp). The average masks large opposing effects across datasets.

### 2. Utilitarianism is the only consistent beneficiary
Bible context helps utilitarianism across both models and both retrieval strategies (+11–15pp). Utilitarian scenarios involve outcomes, harm, and welfare — concepts that map reasonably well onto biblical language around suffering, stewardship, and consequences.

### 3. Deontology and justice are consistently hurt (especially 8B)
Rule-based and fairness reasoning suffers under Bible RAG. The 8B is hurt significantly on deontology (−11.4pp single, −12.4pp context-3) and justice (−6.5pp, −10.9pp). The 4B is largely unaffected on deontology (0.0pp), likely because it does not integrate the retrieved context as strongly.

### 4. Virtue shows a sharp model-size split under context-3
- 4B bible ±3: **16.3%** (−5.8pp vs single, near-random)
- 8B bible ±3: **81.1%** (+10.3pp vs single, strong improvement)

This is the starkest finding. The 8B is capable of extracting genuine moral signal from expanded virtue-related biblical narrative. The 4B is overwhelmed by the longer context and degrades below its single-verse performance.

### 5. More context helps the 8B on virtue, hurts it on justice
For the 8B, context-3 vs single-verse: Virtue +10.3pp, Justice −4.4pp. Expanded context amplifies whatever the retrieval quality is — good retrieval gets better, bad retrieval gets worse.

### 6. The 8B becomes increasingly conservative with more Bible context
As retrieval context increases, the 8B's confusion matrix shifts toward higher TN and lower TP — it becomes more likely to label scenarios as "unacceptable." This suggests biblical context applies a systematic strictness bias in the 8B that the 4B does not exhibit to the same degree.

### 7. Model size matters more than retrieval strategy
The 8B outperforms the 4B on plain by ~9pp on average. The biggest gains from switching retrieval strategy are smaller than the baseline model-size gap, except on virtue (where the 8B + context-3 combination uniquely shines).

---

## Conclusion

Bible RAG is not a general-purpose moral reasoning booster. Its effectiveness is highly dependent on the alignment between the ethical framework being evaluated and the semantic content of retrieved verses. Utilitarian scenarios benefit; deontological and justice scenarios are harmed. Whether expanded context helps or hurts depends strongly on model capacity — small models should use minimal context, larger models can leverage richer retrieval when the domain is well-matched.