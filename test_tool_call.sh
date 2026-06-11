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
  }' | python3 - <<'PY'
import sys, json
try:
    data = json.load(sys.stdin)
except Exception as e:
    print("Could not parse response:", e); sys.exit(1)

msg = (data.get("choices") or [{}])[0].get("message", {})
print(json.dumps(msg, indent=2))
print("\n─────────────────────────────────────────────")

tc = msg.get("tool_calls")
content = (msg.get("content") or "").strip()
looks_textual = any(k in content for k in ("get_weather", "<tool_call", '"name"', "function"))

if tc:
    print("VERDICT: ✅ STRUCTURED tool_calls returned.")
    print("         The server+model are fine — the issue is on the client side.")
elif looks_textual:
    print("VERDICT: ⚠️  Tool call came back as TEXT inside content (not structured).")
    print("         Fix: try `qwen`, and/or add the client fallback parser.")
else:
    print("VERDICT: ❌ No tool call at all — model just chatted.")
    print("         Fix: switch to a stronger tool-calling model (`mlxctl serve qwen`).")
PY