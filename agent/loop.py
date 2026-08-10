import json
import yaml
import time
import os
from groq import Groq
from agent.tools import TOOL_SCHEMAS, ToolExecutor
from agent.prompts import SYSTEM_PROMPT, build_user_prompt
from agent.memory import EpisodicMemory
from perception.scene_state import SceneState
from utils.logger import get_logger

logger = get_logger("agent_loop")


class AgentLoop:
    def __init__(
        self,
        config_path: str = "configs/agent_config.yaml",
        scene_state: SceneState = None,
        client=None,
    ):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.model_cfg = cfg["model"]
        self.loop_cfg = cfg["loop"]

        if client is None and not os.getenv("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY is not configured. Copy .env.example to .env and set it.")
        self.client = client or Groq()  # reads GROQ_API_KEY from env
        self.scene_state = scene_state
        self.executor = ToolExecutor(scene_state=scene_state)
        self.memory = EpisodicMemory(
            max_events=cfg["memory"]["max_events"],
            summarize_after=cfg["memory"]["summarize_after"],
        )

        self._batch_id = 0
        self._frame_counter = 0
        self._cooldown_remaining = 0

    def should_run(self) -> bool:
        self._frame_counter += 1
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return False
        return self._frame_counter % self.loop_cfg["frame_batch_size"] == 0

    def run(self) -> dict:
        self._batch_id += 1
        scene = self.scene_state.get_snapshot()
        for event in self.scene_state.consume_events():
            self.memory.add_scene_event(event)
        memory_ctx = self.memory.get_context()
        user_msg = build_user_prompt(scene, memory_ctx, self._batch_id)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        # Convert Anthropic tool schema format → Groq/OpenAI format
        groq_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in TOOL_SCHEMAS
        ]

        tool_calls_made = []
        tool_results = []
        decision_text = ""
        iterations = 0

        while iterations < self.loop_cfg["max_iterations"]:
            iterations += 1

            try:
                response = self.client.chat.completions.create(
                    model=self.model_cfg["name"],
                    max_tokens=self.model_cfg["max_tokens"],
                    temperature=self.model_cfg["temperature"],
                    tools=groq_tools,
                    tool_choice="auto",
                    messages=messages,
                )
            except Exception as exc:
                raise RuntimeError(f"LLM request failed on agent batch {self._batch_id}: {exc}") from exc

            msg = response.choices[0].message
            messages.append(msg)  # append assistant message

            # Extract DECISION from text
            if msg.content and "DECISION:" in msg.content:
                decision_text = msg.content.split("DECISION:")[-1].strip()

            # No tool calls → done
            if not msg.tool_calls:
                break

            # Execute each tool call
            for tc in msg.tool_calls:
                tool_name = tc.function.name
                try:
                    tool_input = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    tool_input = {}
                    result_str = json.dumps({"error": "LLM returned invalid JSON tool arguments"})
                else:
                    result_str = self.executor.execute(tool_name, tool_input)

                logger.debug(f"Tool call: {tool_name}({tool_input})")
                tool_calls_made.append({"tool": tool_name, "input": tool_input})
                tool_results.append({"tool": tool_name, "result": json.loads(result_str)})
                self.memory.add_tool_result(tool_name, result_str)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str,
                })

        if decision_text:
            self.memory.add_decision(decision_text)

        if tool_calls_made:
            self._cooldown_remaining = self.loop_cfg["cooldown_frames"]

        return {
            "batch_id": self._batch_id,
            "iterations": iterations,
            "tool_calls": tool_calls_made,
            "tool_results": tool_results,
            "decision": decision_text,
            "alerts": self.executor._alerts[-3:],
            "annotations": self.executor._annotations,
        }
