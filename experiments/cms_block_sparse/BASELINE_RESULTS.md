# CMS Block-Sparse Baseline Results

**Date**: 2025-12-26 (Updated)
**Model**: TitanMAC 59.3M params (d_model=640, n_layers=12, n_heads=10)
**Training**: 10K steps per task, batch_size=8, seq_length=256
**Optimizer**: AdamW, lr=3e-4

## Executive Summary

Dense sequential training causes **severe catastrophic forgetting** across all test types:

| Experiment | Type | Task A Accuracy | After Task B | Forgetting | Status |
|------------|------|-----------------|--------------|------------|--------|
| NLP → Math | OOD | 100% | 0% | **100%** | ✅ Valid |
| MODE:STD → Mixed | ID-Semantic | 67.2% | 68.4% | **-1.8%** | ✅ Valid |
| SVO → OVS | ID-Syntactic | 100% | 11.2% | **88.8%** | ✅ Valid |
| MODE:STD → MOD7 | ID-Context | 28.8% | 1.2% | **95.8%** | ✅ Valid |

> **Note**: ID-Semantic and ID-Syntactic experiments were redesigned for rigor. See details below.

---

## Experiment 1: OOD Forgetting (NLP → Math)

### Description

**Out-of-Distribution (OOD)** forgetting tests whether the model can retain knowledge of one domain when trained on a completely different domain. This is the **easiest** test for topology-based solutions because the vocabularies are disjoint.

- **Task A**: Synthetic NLP patterns (antonyms, sequences, POS tagging)
- **Task B**: Standard arithmetic (addition, subtraction, multiplication)
- **Token Overlap**: ~0% (completely different vocabularies)

### Why This Matters

If block-sparse topology can't solve OOD forgetting, something is fundamentally broken. The model should be able to activate different input columns for different vocabularies, keeping math and NLP pathways separate.

### Data Examples

**Task A (NLP) - Training Data:**
```
hot is to cold as big is to: small
A B A B A: B
the quick fox jumps | quick: ADJ
the ___ barks loudly | dog
old is to young as slow is to: quick
```

**Task B (Math) - Training Data:**
```
45 + 32 = 77
123 - 45 = 78
12 * 7 = 84
89 + 11 = 100
56 - 23 = 33
```

### Results

| Metric | Value |
|--------|-------|
| Task A Accuracy (after NLP training) | 100% |
| Task A Accuracy (after Math training) | 0% |
| **Forgetting** | **100%** |

### Interpretation

The model learned NLP patterns perfectly (100% accuracy), then **completely forgot** them after math training. This establishes the upper bound for forgetting - dense sequential training provides no protection whatsoever when domains are different.

---

## Experiment 2: ID-Semantic Forgetting (MODE:STD → Mixed)

### Description

**In-Distribution Semantic** forgetting tests whether the model can learn to distinguish between two interpretations of the same inputs when both are presented with explicit mode markers.

- **Task A**: MODE:STD only (standard arithmetic with mode prefix)
- **Task B**: Mixed interleaved (MODE:STD + MODE:MOD7 in same training)
- **Token Overlap**: ~99% (same numbers, different mode prefix)

### Why This Matters (Design Rationale)

The **original design** (Task A = standard, Task B = modular only) was flawed:
- Training exclusively on modular arithmetic obviously overwrites standard knowledge
- This doesn't test whether topology can maintain separate pathways
- It just confirms "new data replaces old data" (trivial result)

The **new design** tests something meaningful:
- Task A teaches MODE:STD interpretation
- Task B teaches BOTH modes simultaneously
- Question: Can the model learn to distinguish modes when both are present?

### Data Examples

**Task A (MODE:STD only) - Training Data:**
```
MODE:STD | 5 + 3 = 8
MODE:STD | 12 * 4 = 48
MODE:STD | 67 - 23 = 44
MODE:STD | 8 + 9 = 17
MODE:STD | 45 * 2 = 90
```

**Task B (Mixed) - Training Data:**
```
MODE:STD | 45 + 32 = 77
MODE:MOD7 | 12 * 4 = 6
MODE:STD | 89 - 23 = 66
MODE:MOD7 | 8 + 9 = 3
MODE:STD | 15 * 3 = 45
MODE:MOD7 | 5 + 3 = 1
```

**Evaluation:**
```
Input: "MODE:STD | 5 + 3 ="
Expected: "8"

Input: "MODE:MOD7 | 5 + 3 ="
Expected: "1"
```

### Results

| Metric | Value |
|--------|-------|
| MODE:STD Accuracy (after Task A) | 67.2% |
| MODE:STD Accuracy (after Task B) | 68.4% |
| MODE:MOD7 Accuracy (after Task B) | 25.6% |
| **ID Forgetting** | **-1.8%** |

### Interpretation

The negative forgetting (-1.8%) indicates that learning the mixed mode dataset **did not harm** MODE:STD performance. In fact, MODE:STD accuracy slightly improved from 67.2% to 68.4%.

This result is expected because:
1. Task B (mixed) includes MODE:STD examples alongside MODE:MOD7
2. The model continues to see MODE:STD data during Task B training
3. This is **not catastrophic forgetting** because both modes are trained together

Key observations:
- MODE:STD accuracy is moderate (67-68%) indicating the model can learn mode-prefixed arithmetic
- MODE:MOD7 accuracy is lower (25.6%) after Task B, suggesting modular arithmetic is harder
- The negative forgetting validates the experiment design: mixed training preserves prior knowledge

This experiment tests a different question than pure forgetting: "Can the model learn to distinguish modes when both are present?" The answer is partially yes - the model maintains MODE:STD performance while learning MODE:MOD7.

---

## Experiment 3: ID-Syntactic Forgetting (SVO → OVS)

### Description

**In-Distribution Syntactic** forgetting tests whether the model can remember word order semantics. The **same words** in **different grammatical structures** identify different subjects.

- **Task A**: SVO (Subject-Verb-Object) - Active voice: "the cat chased the mouse"
- **Task B**: OVS (Object-Verb-Subject) - Passive voice: "the mouse was chased by the cat"
- **Token Overlap**: 100% (same nouns, same verb roots)

### Why This Matters (Design Rationale)

The **original design** was linguistically invalid:
- "OVS" data was just SVO sentences with nouns swapped
- "the mouse chased the cat | SUBJ: cat" is semantically confusing
- In English, "the mouse chased the cat" clearly means mouse is doing the chasing

The **new design** uses proper linguistic structure:
- SVO uses **active voice**: "the cat chased the mouse" → first noun is subject
- OVS uses **passive voice**: "the mouse was chased by the cat" → last noun is subject
- Same semantic events, same vocabulary, different grammatical structure
- Unambiguous because "by X" clearly marks the agent

### Data Examples

**Task A (SVO - Active Voice) - Training Data:**
```
the cat chased the mouse | SUBJ: cat
the dog found the bird | SUBJ: dog
the fox followed the deer | SUBJ: fox
the wolf caught the frog | SUBJ: wolf
the bear saw the fish | SUBJ: bear
```

**Task B (OVS - Passive Voice) - Training Data:**
```
the mouse was chased by the cat | SUBJ: cat
the bird was found by the dog | SUBJ: dog
the deer was followed by the fox | SUBJ: fox
the frog was caught by the wolf | SUBJ: wolf
the fish was seen by the bear | SUBJ: bear
```

**Evaluation:**
```
# After SVO training:
Input: "the cat chased the mouse |"
Expected: "cat" (first noun is subject in active voice)

# After OVS training:
Input: "the mouse was chased by the cat |"
Expected: "cat" (noun after "by" is subject in passive voice)
```

### Vocabulary Consistency

Both datasets use identical word pools:
- **Nouns**: dog, cat, fox, bird, mouse, bear, fish, frog, wolf, deer
- **Verbs**: chased/chased, followed/followed, watched/watched, helped/helped, found/found, caught/caught, saw/seen, heard/heard

### Results

| Metric | Value |
|--------|-------|
| SVO Accuracy (after Task A) | 100% |
| SVO Accuracy (after Task B) | 11.2% |
| OVS Accuracy (after Task B) | ~100%* |
| **ID Forgetting** | **88.8%** |

*OVS accuracy inferred from very low Task B loss (0.0002)

### Interpretation

This is **severe catastrophic forgetting** (88.8%). The model:
1. Learned SVO perfectly (100% accuracy)
2. Then learned OVS perfectly (loss dropped to 0.0002)
3. Almost completely forgot SVO (only 11.2% retained)

The 11.2% retention is likely random guessing since there are ~10 possible nouns.

This demonstrates that even with the same vocabulary and similar sentence structures, learning a new syntactic pattern catastrophically overwrites the old one. The model cannot maintain separate position-based interpretations for different sentence structures.

This is the expected baseline behavior for dense networks. Block-sparse topology should be able to maintain separate pathways for:
- "the X verbed the Y | SUBJ:" → X (active voice pattern)
- "the Y was verbed by the X | SUBJ:" → X (passive voice pattern)

---

## Experiment 4: ID-Context Forgetting (MODE:STD → MODE:MOD7)

### Description

**In-Distribution Context** forgetting tests whether a **single prefix token** can route to entirely different computations. This is the **hardest** test.

- **Task A**: MODE:STD prefix → standard arithmetic
- **Task B**: MODE:MOD7 prefix → modular arithmetic
- **Token Overlap**: ~99% (only the MODE token differs)

### Why This Matters

This is explicit mode-switching. A single token must control the entire computational pathway. If topology can solve this, it has learned true **mode-awareness** - the holy grail of continual learning.

### Data Examples

**Task A (MODE:STD) - Training Data:**
```
MODE:STD | 5 + 3 = 8
MODE:STD | 12 * 4 = 48
MODE:STD | 67 - 23 = 44
MODE:STD | 8 + 9 = 17
MODE:STD | 45 * 2 = 90
```

**Task B (MODE:MOD7) - Training Data:**
```
MODE:MOD7 | 5 + 3 = 1
MODE:MOD7 | 12 * 4 = 6
MODE:MOD7 | 67 - 23 = 2
MODE:MOD7 | 8 + 9 = 3
MODE:MOD7 | 45 * 2 = 6
```

**Evaluation:**
```
Input: "MODE:STD | 5 + 3 ="
Standard Expected: "8"

Input: "MODE:STD | 12 * 4 ="
Standard Expected: "48"
```

### Results

| Metric | Value |
|--------|-------|
| Standard Accuracy (after Task A) | 28.8% |
| Standard Accuracy (after Task B) | 1.2% |
| Modular Accuracy (after Task B) | 12.8% |
| **ID Forgetting** | **95.8%** |

### Interpretation

The MODE prefix task is **difficult to learn** (only 28.8% on standard). The model struggles to use the prefix token as a routing signal. After MOD7 training:
- Standard accuracy crashed to 1.2% (95.8% forgetting)
- Modular accuracy is only 12.8% (didn't learn Task B well either)

This confirms that explicit mode-switching is the hardest challenge. Dense networks cannot maintain separate pathways for mode-switched inputs.

---

## Success Criteria for Block-Sparse Topology

Based on these baselines, the CMS Dynamic Block-Sparse layer should achieve:

| Experiment | Baseline Forgetting | Target Forgetting | Tier |
|------------|---------------------|-------------------|------|
| OOD (NLP→Math) | 100% | < 15% | Gold |
| ID-Syntactic (SVO→OVS) | 88.8% | < 30% | Silver |
| ID-Semantic (Mixed) | -1.8%* | N/A | Reference |
| ID-Context (MODE) | 95.8% | < 50% | Stretch |

*Negative forgetting in ID-Semantic indicates no forgetting because Task B includes Task A data.

### Recommended Test Priority

1. **OOD (NLP→Math)** - 100% forgetting baseline, should be solved first
2. **ID-Syntactic (SVO→OVS)** - 88.8% forgetting, tests position-dependent semantics
3. **ID-Context (MODE)** - 95.8% forgetting, hardest (mode-prefix routing)
4. **ID-Semantic (Mixed)** - Reference only (no forgetting due to design)

---

## Reproduction Commands

```bash
# OOD baseline (swap tasks so NLP is Task A)
python experiments/cms_block_sparse/run_baselines.py \
  --experiment ood_math_nlp --baseline dense_sequential \
  --steps 10000 --device cuda --wandb --swap-tasks

# ID-Semantic baseline (NEW: uses mixed mode in Task B)
python experiments/cms_block_sparse/run_baselines.py \
  --experiment id_semantic_modular --baseline dense_sequential \
  --steps 10000 --device cuda --wandb

# ID-Syntactic baseline (NEW: uses passive voice OVS)
python experiments/cms_block_sparse/run_baselines.py \
  --experiment id_syntactic_grammar --baseline dense_sequential \
  --steps 10000 --device cuda --wandb

# ID-Context baseline
python experiments/cms_block_sparse/run_baselines.py \
  --experiment id_context_mode --baseline dense_sequential \
  --steps 10000 --device cuda --wandb
```

---

## Changelog

### 2025-12-26 (v3)
- **ID-Syntactic baseline completed**: 88.8% forgetting
  - Model learns SVO perfectly (100%), then OVS perfectly (loss 0.0002)
  - SVO accuracy drops to 11.2% (catastrophic forgetting confirmed)
- **ID-Semantic baseline completed**: -1.8% forgetting
  - Mixed mode training preserves MODE:STD performance (67.2% → 68.4%)
  - This is expected since Task B includes Task A data
  - Result validates experiment design, not useful as forgetting benchmark
- **Fixed vocabulary size bug**: Changed from 89 to 100 to include all NLP tokens
- **Fixed benchmark evaluation**: Added proper evaluation formats for SVO, OVS, MODE:STD, MODE:MOD7

### 2025-12-26 (v2)
- **ID-Semantic**: Redesigned from sequential replacement to mixed mode training
  - Task B now includes both MODE:STD and MODE:MOD7 interleaved
  - Tests whether model can learn to distinguish modes, not just overwrite
- **ID-Syntactic**: Fixed linguistically invalid OVS data
  - SVO now uses active voice past tense: "the cat chased the mouse"
  - OVS now uses passive voice: "the mouse was chased by the cat"
  - Same vocabulary in both (nouns + transitive verbs)
  - Grammatically correct and semantically unambiguous

### 2025-12-26 (v1)
- Initial baseline measurements for all 4 experiment types

---

## wandb Runs

All experiments logged to: https://wandb.ai/adew-me/cms-block-sparse

Key metrics tracked:
- `forgetting/accuracy_a_after_task_a` - Task A accuracy before Task B
- `forgetting/accuracy_a_after_task_b` - Task A accuracy after Task B
- `forgetting/id_forgetting_pct` - ID-specific forgetting percentage
- `forgetting/standard_accuracy_*` - Standard task accuracy
- `forgetting/modular_accuracy_*` - Modular/alternate task accuracy
