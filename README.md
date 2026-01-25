# prompt-lab

Test prompt variants across LLM providers with LLM-as-judge evaluation.

## Installation

```bash
uv sync
```

Create `.env` with your API keys:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

## Quick Start

```bash
# Run an experiment
prompt-lab run experiments/my-experiment/v1

# View results
prompt-lab results experiments/my-experiment/v1

# Compare variants
prompt-lab compare experiments/my-experiment
```

## How It Works

```
prompt.md + inputs.yaml → LLM → response → judge.md → score
```

1. **prompt.md** is the subject under test (the prompt you want to evaluate)
2. **inputs.yaml** provides test cases with variables for the prompt
3. The prompt is sent to each configured **model** (LLM)
4. **judge.md** evaluates each response and assigns a score

Create multiple variants (v1, v2, etc.) to compare different prompt approaches.

## Experiment Structure

```
my-experiment/
├── experiment.md       # Config: models, runs (required)
├── judge.md            # Evaluator: scoring criteria (required)
├── inputs.yaml         # Shared test cases (optional, used by all variants)
├── v1/                 # Variant (at least one required)
│   ├── prompt.md       # ⭐ Subject under test (required)
│   ├── inputs.yaml     # Variant-specific inputs (overrides experiment-level)
│   └── tools.yaml      # Tool definitions (optional)
└── v2/                 # Another variant to compare...
    └── prompt.md       # ⭐ Different prompt approach
```

Both `judge.md` and `inputs.yaml` support fallback: if not found in the variant folder, the experiment-level file is used. This allows sharing test cases across variants for fair A/B comparison.

## File Formats

### experiment.md

Defines the experiment name, description, models, and default number of runs per input.

```yaml
---
name: my-experiment
description: Testing different prompt styles
hypothesis: Concise prompts will score higher than verbose ones
models:
  - openai:gpt-4o-mini
  - anthropic:claude-sonnet-4-20250514
runs: 5
---

Optional markdown content describing the experiment.
```

**Experiment options:**

| Option | Default | Description |
|--------|---------|-------------|
| `name` | folder name | Experiment identifier |
| `description` | `""` | Brief description |
| `models` | required | List of models to test |
| `runs` | `5` | Runs per input (for statistical analysis) |
| `hypothesis` | `""` | What you're testing (displayed in results) |

### prompt.md (subject under test)

The prompt you want to evaluate. This is what gets sent to the LLM. Use `{{ variables }}` to inject test data from `inputs.yaml`.

```yaml
---
description: Friendly greeting style
---

You are a friendly assistant. Greet the user warmly.

User: {{ name }}
```

Each variant folder contains a different `prompt.md` to compare approaches (e.g., formal vs casual tone, different instructions, etc.).

**Prompt options:**

| Option | Default | Description |
|--------|---------|-------------|
| `models` | experiment models | Override which models to test for this variant |

Example with models override:

```yaml
---
description: GPT-4 optimized prompt
models:
  - openai:gpt-4o
---

You are a helpful assistant...
```

### inputs.yaml (optional)

Test cases with variables matching the prompt template. If omitted, runs once with empty data (useful for static prompts without variables).

```yaml
- id: alice
  name: Alice

- id: bob
  name: Bob
  runs: 10  # Override experiment's runs for this input
```

**Location:** Can be placed at experiment level (shared across all variants) or in a variant folder (variant-specific). Variant-level inputs take precedence over experiment-level.

**Input options:**

| Field | Default | Description |
|-------|---------|-------------|
| `id` | `input-N` | Unique identifier for results |
| `runs` | experiment runs | Override runs for this specific input |
| (other) | - | Variables available in prompt template |

### tools.yaml (optional)

Define tools (functions) that the LLM can call during execution. Useful for testing prompts that involve function calling.

```yaml
- name: get_weather
  description: Get current weather for a location
  parameters:
    type: object
    properties:
      location:
        type: string
        description: City name
      unit:
        type: string
        enum: [celsius, fahrenheit]
    required:
      - location

- name: search
  description: Search the web
  parameters:
    type: object
    properties:
      query:
        type: string
```

**Tool fields:**

| Field | Required | Description |
|-------|----------|-------------|
| `name` | yes | Tool/function name |
| `description` | no | What the tool does |
| `parameters` | no | JSON Schema for tool parameters |

Tool calls made by the model are captured in the response and available for judge evaluation.

### judge.md (evaluator)

Defines how to score each LLM response. The judge is another LLM that evaluates quality based on your criteria.

```yaml
---
model: openai:gpt-4o-mini
score_range: [1, 10]
temperature: 0
---

You are evaluating a greeting response.

## Rubric
- **10**: Uses user's name, warm tone, offers to help
- **8-9**: Uses name and friendly, but generic
- **6-7**: Friendly but doesn't use name
- **4-5**: Cold or overly formal
- **1-3**: Inappropriate or ignores user

**User Input:** {{ user_input }}
**Model Response:** {{ response }}
```

**Judge options:**

| Option | Default | Description |
|--------|---------|-------------|
| `model` | `openai:gpt-4o` | Model to use for judging |
| `score_range` | `[1, 10]` | Min and max score |
| `temperature` | `0` | 0 = deterministic, higher = more varied |

## Multiple Runs & Statistics

For more reliable evaluation, run each input multiple times and get statistical analysis:

```yaml
# experiment.md
---
name: my-experiment
models:
  - openai:gpt-4o-mini
runs: 5
---
```

Results show hypothesis and mean with 95% confidence interval:

```
Hypothesis: Concise prompts will score higher than verbose ones

┏━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Input ┃ Model              ┃ Mean ┃ 95% CI       ┃ Range ┃ Scores        ┃
┡━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━┩
│ alice │ openai:gpt-4o-mini │ 9.2  │ (8.5-9.9)    │ 8-10  │ 9, 10, 9, 9, 9│
│ bob   │ openai:gpt-4o-mini │ 8.4  │ (7.8-9.0)    │ 8-9   │ 8, 9, 8, 8, 9 │
└───────┴────────────────────┴──────┴──────────────┴───────┴───────────────┘

⚠ Low sample size (3 runs). Consider runs: 5+ for reliable statistics.
```

When `runs > 1`:
- Cache is disabled to get independent LLM responses
- Each input is evaluated N times
- 95% confidence intervals show the reliability of your results
- Warning shown when sample size is too small for reliable statistics

## Statistical Significance

When comparing variants, the `compare` command shows whether differences are statistically significant:

```bash
prompt-lab compare experiments/my-experiment
```

```
Hypothesis: Concise prompts will score higher than verbose ones

┏━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━┓
┃ Variant ┃ Mean Score ┃ 95% CI       ┃ Avg Latency ┃ Runs ┃
┡━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━┩
│ v1      │ 8.5/10     │ (8.1-8.9)    │ 450ms       │ 2×5  │
│ v2      │ 7.2/10     │ (6.8-7.6)    │ 420ms       │ 2×5  │
└─────────┴────────────┴──────────────┴─────────────┴──────┘

Statistical Significance (Welch's t-test, α=0.05):

  ✓ v1 > v2 (p≤0.01)
```

This helps you know if v1 is actually better than v2, or if the difference is just noise.

## Templating

Variables from `inputs.yaml` are available in prompts:

```markdown
Hello {{ name }}, you are {{ age }} years old.
```

Literal braces don't need escaping:

```markdown
Return JSON: {"result": "value"}
```

## CLI Commands

### run

Run a prompt experiment.

```bash
# Run single variant
prompt-lab run experiments/my-experiment/v1

# Run all variants
prompt-lab run experiments/my-experiment --all

# Run specific model only
prompt-lab run experiments/my-experiment/v1 --model openai:gpt-4o-mini

# Skip cache (fresh API calls)
prompt-lab run experiments/my-experiment/v1 --no-cache
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--model` | `-m` | Run only this model |
| `--all` | `-a` | Run all variants in experiment |
| `--no-cache` | | Disable response caching |

### results

Show results table for a variant.

```bash
prompt-lab results experiments/my-experiment/v1

# Show specific run
prompt-lab results experiments/my-experiment/v1 --run 2026-01-25T19-30-00
```

### compare

Compare results across all variants.

```bash
prompt-lab compare experiments/my-experiment
```

### show

Show detailed responses with judge reasoning.

```bash
# Show all responses
prompt-lab show experiments/my-experiment/v1

# Filter by input
prompt-lab show experiments/my-experiment/v1 --input alice

# Filter by model
prompt-lab show experiments/my-experiment/v1 --model openai:gpt-4o-mini

# Combine filters
prompt-lab show experiments/my-experiment/v1 --input alice --model openai:gpt-4o-mini
```

### cache

Manage response cache.

```bash
# Clear all cached responses
prompt-lab cache clear
```

## Supported Providers

| Provider | Model format |
|----------|--------------|
| OpenAI | `openai:gpt-4o`, `openai:gpt-4o-mini` |
| Anthropic | `anthropic:claude-sonnet-4-20250514` |

## License

MIT
