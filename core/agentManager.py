from typing import List, Dict, Any, Callable
from tools.workloadModel import predictWorkload

class AgentManager:
    def __init__(self, agents: List[Any]):
        self.agents = agents
        self.context: List[Dict[str, Any]] = []
        self.toolRegistry: Dict[str, Callable] = {
            "workloadPredictor": predictWorkload,
        }

    def handleToolCall(self, toolName: str, args):
        if toolName not in self.toolRegistry:
            return {"error": f"Unknown tool '{toolName}'"}

        toolFn = self.toolRegistry[toolName]

        # --- VALIDATION PATCH ---
        if not isinstance(args, (list, tuple)):
            return {"error": "Tool args must be a list"}

        clean_args = []
        for a in args:
            try:
                clean_args.append(float(a))
            except:
                return {"error": f"Invalid numeric value for workloadPredictor: {a}"}

        return toolFn(clean_args)

    def process(self, userRequest: str) -> Any:
        message: Any = userRequest

        for agent in self.agents:
            print(f"\n--- RUNNING {agent.name} ---")
            print(f"input message: {message}\n")

            agentOutput = agent.run(message, self.context)

            outputType = agentOutput.get("type")
            if outputType == "toolCall":
                toolName = agentOutput.get("toolName")
                args = agentOutput.get("args", [])
                toolResult = self.handleToolCall(toolName, args)

                self.context.append(
                    {
                        "agent": agent.name,
                        "toolName": toolName,
                        "toolResult": toolResult,
                    }
                )
                message = {
                    "toolName": toolName,
                    "toolResult": toolResult,
                }

            elif outputType == "message":
                content = agentOutput.get("content")
                self.context.append(
                    {
                        "agent": agent.name,
                        "message": content,
                    }
                )
                message = content

            else:
                # fallback if LLM misbehaves
                self.context.append(
                    {
                        "agent": agent.name,
                        "raw": agentOutput,
                    }
                )
                message = agentOutput
           
            print(f"output from {agent.name}: {agentOutput}\n")
        return message
