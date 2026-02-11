---
model: openai:gpt-4o-mini
score_range: [1, 10]
temperature: 0
---

You are evaluating product names for brandability and consistency. Be fair but critical.

## Scoring (start at 6, add/subtract)

### Format Consistency (-2 to +2)
- **+2**: ALL names follow identical pattern (all CamelCase compounds OR all two-word)
- **+1**: Minor inconsistency (4/5 match)
- **0**: Some inconsistency (3/5 match)
- **-1**: Mixed formats throughout
- **-2**: Chaotic formatting (random styles)

### Brandability (-1 to +2)
- **+2**: All names are single CamelCase compounds (FlexiFit, OmniStep)
- **+1**: All names are clean two-word phrases (Flexi Fit)
- **0**: Mix of compound and multi-word
- **-1**: Names with 3+ words or verbose phrases

### Seed Integration (-1 to +1)
- **+1**: Direct, clever use of seed words in most names
- **0**: Conceptual connection but seeds not directly used
- **-1**: Generic names that ignore seeds entirely

### Memorability (-1 to +1)
- **+1**: Punchy, easy to say, would work as hashtag
- **0**: Acceptable but forgettable
- **-1**: Awkward, hard to pronounce, or too long

## Soft Penalties (apply judiciously)
- Names including product category ("Shoes", "Footwear"): **-0.5 per occurrence** (max -1)
- More than 2 words in any single name: **-0.5 per occurrence** (max -1)

## Score Guide
- **9-10**: Exceptional - perfect consistency, all brandable compounds, creative
- **7-8**: Good - consistent format, mostly brandable, uses seeds well
- **5-6**: Average - some inconsistency or generic names
- **3-4**: Below average - mixed formats, verbose, weak seed connection
- **1-2**: Poor - chaotic, unprofessional, ignores requirements

## Evaluation Context
**Prompt:** {{ prompt }}
**Generated Names:** {{ response }}

Calculate final score (1-10) with brief justification.
