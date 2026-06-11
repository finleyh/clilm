#!/bin/sh
# Quick check: does the LLM endpoint return STRUCTURED tool calls?
# Usage:  sh test_tools.sh                         (defaults to localhost:8080, model "default")
#         sh test_tools.sh http://10.0.2.2:8080 qwen
URL="${1:-http://localhost:8080}"
MODEL="${2:-default}"

echo "→ testing $URL/v1/chat/completions  (model: $MODEL)"
echo

RESP=$(curl -s "$URL/v1/chat/completions" \
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
  }')

if [ -z "$RESP" ]; then
  echo "No response from $URL — is the server up and reachable from here?"
  echo "Try: curl -s $URL/v1/models"
  exit 1
fi

printf '%s' "$RESP" | python3 -c '
import sys, json
raw = sys.stdin.read()
try:
    data = json.loads(raw)
except Exception as e:
    print("Could not parse response as JSON:", e)
    print("--- raw response ---")
    print(raw[:2000])
    sys.exit(1)

msg = (data.get("choices") or [{}])[0].get("message", {})
print(json.dumps(msg, indent=2))
print("\n---------------------------------------------")

tc = msg.get("tool_calls")
content = (msg.get("content") or "").strip()
looks_textual = any(k in content for k in ("get_weather", "tool_call", "arguments", "function"))

if tc:
    print("VERDICT: STRUCTURED tool_calls returned.")
    print("         Server+model are fine -- issue is on the client side.")
elif looks_textual:
    print("VERDICT: Tool call came back as TEXT inside content (not structured).")
    print("         The client fallback parser should now recover this.")
else:
    print("VERDICT: No tool call -- model just chatted.")
    print("         Try a stronger tool-calling model: mlxctl serve qwen")
'