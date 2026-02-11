---
model: openai:gpt-4o-mini
score_range: [1, 10]
---

You are evaluating whether an AI assistant correctly used (or didn't use) a weather tool.

## Context

The AI has access to a `get_weather` tool that fetches real weather data.

- **Weather questions** (e.g., "What's the weather in Paris?") → SHOULD call `get_weather`
- **Non-weather questions** (e.g., "Hello", "What's 2+2?") → Should NOT call any tool

## Input Information

**System instructions:** {{ system_prompt }}
**User message:** {{ prompt }}

## AI Response

**Text response:** {{ response }}
**Tool calls made:** {{ tool_calls }}

## Rubric

- **10**: Perfect behavior
  - If should_call_tool=true: Called `get_weather` with correct location
  - If should_call_tool=false: Did NOT call any tool, gave helpful text response

- **7-9**: Mostly correct
  - Called the right tool but minor parameter issues (e.g., "Paris, France" instead of "Paris")
  - Or didn't call tool when optional but would have been helpful

- **4-6**: Partially correct
  - Called tool when shouldn't have, OR
  - Didn't call tool when clearly should have, OR
  - Called wrong tool

- **1-3**: Wrong behavior
  - Completely wrong tool usage
  - Hallucinated weather data instead of calling tool
