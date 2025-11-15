from typing import Any, Dict, List, Callable

class Agent:
    def __init__(self, name: str, llm: Callable[[str], Dict[str, Any]], tools=None):
        self.name = name
        self.llm = llm
        self.tools = tools or {}

    def run(self, message: Any, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement run()")
