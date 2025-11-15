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

        if isinstance(args, (list, tuple)):
            return toolFn(args)
        else:
            return toolFn([args])

    def process(self, userRequest: str) -> Any:
        message: Any = userRequest

        for agent in self.agents:
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

        return message
