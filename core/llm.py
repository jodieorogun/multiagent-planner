import subprocess, json, re

MODEL_NAME = "qwen2.5-coder:1.5b"

def call_llm(prompt: str) -> dict:
    full_prompt = f"""
You MUST respond ONLY with valid JSON.
Absolutely NO text before or after the JSON block.

{prompt}
"""

    process = subprocess.Popen(
        ["ollama", "run", MODEL_NAME],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    try:
        output, _ = process.communicate(full_prompt.encode("utf-8"), timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        return {"type": "message", "content": "LLM timed out — response skipped."}

    raw = output.decode().strip()

    parsed = extract_json_strong(raw)
    if parsed is not None:
        return parsed

    # LAST RESORT — give the raw text
    return {"type": "message", "content": raw}


def extract_json_strong(text: str):
    """
    Extract the first JSON object from a string, even if the model adds text.
    Fixes minor JSON formatting errors.
    """

    # 1. Find anything that *looks* like JSON object
    candidates = re.findall(r"\{(?:.|\n)*\}", text)

    for c in candidates:
        # 2. Try raw first
        try:
            return json.loads(c)
        except:
            pass

        # 3. Try minor repairs
        fixed = (
            c.replace("'", '"')       # single → double quotes
             .replace("None", "null") # python null
             .replace("Null", "null")
             .replace("NULL", "null")
        )

        try:
            return json.loads(fixed)
        except:
            continue

    return None
