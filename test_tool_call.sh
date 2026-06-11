#!/bin/sh
# Quick check: does the LLM endpoint return STRUCTURED tool calls?
# Usage:  sh test_tools.sh                         (defaults to localhost:8080, model "default")
#         sh test_tools.sh http://10.0.2.2:8080 qwen
URL="${1:-http://localhost:8080}"
MODEL="${2:-default}"

echo "→ testing $URL/v1/chat/completions  (model: $MODEL)"
echo

curl -s "$URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer local" \
  -d '{
    "model": "'"$MODEL"'",
    "stream": false,
    "tool_choice": "auto",
    "messages": [{"role":"user","content":"What is the weather in Paris? Use the get_weather tool."}],
    "tools": [{"type":"function","function":{
      "name":"get_weather",
      "description":"Get the current weather for a city.",
      "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]
  }' | python3 -m json.tool

echo
echo "─────────────────────────────────────────────"
echo "Look at choices[0].message above:"
echo "  • has a \"tool_calls\" array  -> server emits structured calls (problem is our side)"
echo "  • only \"content\" text/JSON  -> no structured calls (switch to qwen / add fallback parser)"