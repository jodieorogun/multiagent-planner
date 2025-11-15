import json
import subprocess

MODEL_NAME = "mistral" 


def call_llm(prompt: str) -> dict:
    full_prompt = f"""
You are an LLM agent in a multi-agent system.

You MUST respond ONLY with strict JSON.
No explanations. No markdown.

You have two valid formats:

1) Message:
{{
  "type": "message",
  "content": <JSON-serialisable content>
}}

OR

2) Tool call:
{{
  "type": "toolCall",
  "toolName": "<tool_name>",
  "args": [...]
}}

Now respond to this prompt:
{prompt}
"""

    # Launch Ollama subprocess
    process = subprocess.Popen(
        ["ollama", "run", MODEL_NAME],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    output, error = process.communicate(full_prompt.encode("utf-8"))

    result = output.decode().strip()

    # Try to extract clean JSON
    try:
        return json.loads(result)
    except Exception:
        # fallback: find the first { and last }
        start = result.find("{")
        end = result.rfind("}") + 1
        cleaned = result[start:end]
        return json.loads(cleaned)
