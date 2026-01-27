---
name: weather-tool-calling
description: Test if AI correctly uses the weather tool when appropriate
hypothesis: AI should call get_weather for weather questions, not for other queries
models:
  - openai:gpt-4o-mini
runs: 3
---

This experiment tests whether the AI knows WHEN to use a tool and WHEN to just respond normally.
