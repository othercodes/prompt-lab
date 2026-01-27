# Enhancement Plan: Improving LLM-as-Judge Evaluation

## Overview

Four enhancements to reduce bias and improve evaluation accuracy in prompt-lab.

---

## 1. Chain-of-Thought in Judge (Priority: HIGH)

**Goal**: Force judge to explain reasoning before scoring, improving alignment with human judgment.

**Current State**:
- Judge returns `score` and optional `reasoning`
- No enforcement of reasoning-before-score pattern

**Implementation**:

### 1.1 Update judge prompt template
```markdown
## Instructions
1. First, analyze the response step by step
2. Consider each rubric criterion
3. Only after analysis, provide your final score

## Output Format
Reasoning: [Your step-by-step analysis]
Score: [number]
```

### 1.2 Update judge response parsing
- File: `promptlab/infrastructure/providers/base.py` or provider implementations
- Ensure `reasoning` is captured from CoT output
- Parse score from structured output

### 1.3 Add `cot` option to judge.md
```yaml
---
model: openai:gpt-4o-mini
score_range: [1, 10]
temperature: 0
chain_of_thought: true  # NEW: enforce CoT
---
```

### 1.4 Files to modify
- `promptlab/infrastructure/yaml_config_loader.py` - parse `chain_of_thought` option
- `promptlab/domain/contracts/config.py` - add field to `JudgeConfig`
- `promptlab/application/run_experiment.py` - apply CoT prompt wrapper
- `tests/test_loader.py` - add tests

**Effort**: Low (1-2 hours)

---

## 2. Multi-Judge Support (Priority: MEDIUM)

**Goal**: Use multiple judge models and aggregate scores to reduce self-enhancement bias.

**Current State**:
- Single judge model per experiment
- No score aggregation

**Implementation**:

### 2.1 Update judge.md to support multiple models
```yaml
---
models:  # NEW: list of judge models
  - openai:gpt-4o-mini
  - anthropic:claude-sonnet-4-20250514
aggregation: mean  # mean | median | majority
score_range: [1, 10]
---
```

### 2.2 Create aggregation logic
- File: `promptlab/domain/judge_aggregator.py` (new)
- Strategies: mean, median, majority vote
- Return aggregated score + individual scores

### 2.3 Update result schema
```python
@dataclass
class JudgeResult:
    score: float  # aggregated
    reasoning: str
    individual_scores: list[dict]  # NEW: per-judge scores
```

### 2.4 Files to modify
- `promptlab/domain/contracts/config.py` - update `JudgeConfig`
- `promptlab/domain/contracts/results.py` - update `RunResult`
- `promptlab/infrastructure/yaml_config_loader.py` - parse judge models list
- `promptlab/application/run_experiment.py` - run multiple judges
- `promptlab/domain/judge_aggregator.py` - NEW: aggregation logic
- `promptlab/infrastructure/console_display.py` - show individual judge scores
- `tests/test_judge_aggregator.py` - NEW: tests

**Effort**: Medium (3-4 hours)

---

## 3. Position Swap Test (Priority: MEDIUM)

**Goal**: Detect position bias in pairwise comparisons by swapping response order.

**Current State**:
- No pairwise comparison mode
- Single response evaluation only

**Implementation**:

### 3.1 Add pairwise comparison mode
```yaml
# experiment.md
---
name: my-experiment
mode: pairwise  # NEW: single | pairwise
models:
  - openai:gpt-4o-mini
---
```

### 3.2 Create pairwise judge template
```markdown
Compare Response A and Response B.

**Response A:** {{ response_a }}
**Response B:** {{ response_b }}

Which is better? Output: A, B, or TIE
```

### 3.3 Implement position swap
- Run comparison: A vs B → result1
- Run comparison: B vs A → result2
- Flag inconsistency if results differ
- Mark as "stable" or "unstable" preference

### 3.4 New result fields
```python
@dataclass
class PairwiseResult:
    preferred: str  # "v1" | "v2" | "tie"
    position_stable: bool  # same result after swap
    confidence: float  # based on stability
```

### 3.5 Files to modify/create
- `promptlab/domain/contracts/config.py` - add `mode` to ExperimentConfig
- `promptlab/domain/contracts/results.py` - add `PairwiseResult`
- `promptlab/application/run_pairwise.py` - NEW: pairwise runner
- `promptlab/infrastructure/console_display.py` - pairwise display
- `tests/test_pairwise.py` - NEW: tests

**Effort**: Medium-High (4-6 hours)

---

## 4. Gold Standard Calibration (Priority: LOW)

**Goal**: Validate judge accuracy against human-labeled examples.

**Current State**:
- No calibration mechanism
- No way to measure judge reliability

**Implementation**:

### 4.1 Add calibration.yaml format
```yaml
# experiments/my-experiment/calibration.yaml
- input_id: example1
  response: "Hello Alice! How can I help you today?"
  human_score: 9
  notes: "Warm, uses name, offers help"

- input_id: example2
  response: "Hi."
  human_score: 3
  notes: "Too brief, no personalization"
```

### 4.2 Add calibrate command
```bash
prompt-lab calibrate experiments/my-experiment

# Output:
# Judge Calibration Report
# ┏━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━┓
# ┃ Example  ┃ Human Score ┃ Judge Score ┃ Diff ┃
# ┡━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━┩
# │ example1 │ 9           │ 8           │ -1   │
# │ example2 │ 3           │ 4           │ +1   │
# └──────────┴─────────────┴─────────────┴──────┘
#
# Metrics:
#   Pearson correlation: 0.92
#   Mean absolute error: 0.8
#   Agreement (±1): 85%
```

### 4.3 Calibration metrics
- Pearson correlation coefficient
- Mean absolute error (MAE)
- Agreement rate (within ±1 point)
- Cohen's Kappa for categorical agreement

### 4.4 Files to modify/create
- `promptlab/domain/contracts/config.py` - add `CalibrationCase`
- `promptlab/domain/calibration.py` - NEW: calibration logic
- `promptlab/infrastructure/yaml_config_loader.py` - parse calibration.yaml
- `promptlab/infrastructure/console_display.py` - calibration report
- `cli.py` - add `calibrate` command
- `tests/test_calibration.py` - NEW: tests

**Effort**: Medium (3-4 hours)

---

## Implementation Order

| Phase | Enhancement | Effort | Cumulative |
|-------|-------------|--------|------------|
| 1 | Chain-of-Thought | 1-2h | 1-2h |
| 2 | Multi-Judge | 3-4h | 4-6h |
| 3 | Gold Standard Calibration | 3-4h | 7-10h |
| 4 | Position Swap (Pairwise) | 4-6h | 11-16h |

---

## Quick Wins (Can Do Now)

1. **CoT prompt example** - Add to README as best practice
2. **Judge model diversity warning** - Warn if judge model = test model (self-enhancement bias)
3. **Reasoning requirement** - Fail if judge doesn't return reasoning

---

## Success Metrics

After implementation, measure:
- Judge-human agreement rate (target: >80%)
- Score variance reduction with multi-judge
- Position bias detection rate
- Calibration correlation (target: >0.85 Pearson)
