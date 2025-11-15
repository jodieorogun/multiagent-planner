from typing import Any, Dict, List, Callable
from core.llm import call_llm

class BaseAgent:
    def __init__(self, name: str, llm: Callable[[str], Dict[str, Any]] = None, tools=None):
        self.name = name
        self.llm = llm or call_llm
        self.tools = tools or {}

    def run(self, message: Any, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement run()")
