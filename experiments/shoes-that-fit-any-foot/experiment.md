---
name: shoes-that-fit-any-foot
description: Test zero-shot vs few-shot prompts for product name generation
hypothesis: Few-shot prompts with examples will produce more consistent, brandable product names than zero-shot prompts
models:
  - openai:gpt-4o-mini
---

# Shoes That Fit Any Foot Experiment

Compare how the usage of examples in prompts affects product name generation quality.

## Variants
- **v1 (zero-shot)**: No examples provided
- **v2 (few-shot)**: Two product naming examples included

## Success Criteria
- Format consistency (all names follow same pattern)
- Brandability (single CamelCase compounds preferred)
- Seed word integration
