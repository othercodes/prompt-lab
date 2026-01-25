---
model: openai:gpt-4o-mini
score_range: [1, 10]
temperature: 0
---

You are evaluating a greeting response.

## Rubric (use exactly)
- **10**: Uses user's name, warm tone, offers to help - perfect
- **8-9**: Uses name and friendly, but slightly generic
- **6-7**: Friendly but doesn't use name OR generic response
- **4-5**: Cold or overly formal
- **1-3**: Inappropriate or ignores the user

**User Input:** {{ user_input }}
**Model Response:** {{ response }}
