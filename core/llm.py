import json
import os
import subprocess
import sys
from typing import Any, Dict


DEFAULT_MODEL = "qwen2.5:3b"
DEFAULT_TIMEOUT_SECONDS = 45


class LLMError(RuntimeError):
    """Raised when Ollama cannot provide a usable response."""


def extract_json_object(text: str) -> Dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise LLMError("Ollama returned no valid JSON object")


class OllamaClient:
    def __init__(
        self, model: str = None, timeout_seconds: int = None, debug: bool = False
    ):
        self.model = model or os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)
        self.timeout_seconds = timeout_seconds or int(
            os.getenv("OLLAMA_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)
        )
        self.debug = debug

    def complete(self, prompt: str) -> Dict[str, Any]:
        strict_prompt = (
            "Respond with one valid JSON object only. Do not use Markdown fences.\n\n"
            + prompt
        )
        try:
            process = subprocess.run(
                [
                    "ollama",
                    "run",
                    self.model,
                    "--format",
                    "json",
                    "--nowordwrap",
                ],
                input=strict_prompt,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except FileNotFoundError as exc:
            raise LLMError(
                "Ollama is not installed or is not available on PATH"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise LLMError(
                f"Ollama did not respond within {self.timeout_seconds} seconds"
            ) from exc

        if process.returncode != 0:
            detail = process.stderr.strip() or "unknown Ollama error"
            raise LLMError(f"Ollama failed: {detail}")
        if self.debug:
            print(f"[ollama:{self.model}] {process.stdout.strip()}", file=sys.stderr)
        return extract_json_object(process.stdout)
