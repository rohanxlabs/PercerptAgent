from collections import deque
import time


class EpisodicMemory:
    """
    Short-term memory of agent decisions and scene events.
    Summarized when it grows beyond a threshold.
    """

    def __init__(self, max_events: int = 20, summarize_after: int = 15):
        self.max_events = max_events
        self.summarize_after = summarize_after
        self._events: deque = deque(maxlen=max_events)
        self._summary: str = ""
        self._total_added = 0

    def add(self, event_type: str, content: str):
        self._events.append({
            "type": event_type,
            "content": content,
            "timestamp": time.time(),
        })
        self._total_added += 1
        if len(self._events) >= self.summarize_after:
            self._summary = f"{len(self._events)} most recent events retained; newest: {content[:160]}"

    def get_context(self) -> str:
        if not self._events:
            return "No prior memory."

        lines = []
        if self._summary:
            lines.append(f"[Earlier summary]: {self._summary}")

        for e in self._events:
            t = time.strftime("%H:%M:%S", time.localtime(e["timestamp"]))
            lines.append(f"  [{t}] {e['type']}: {e['content']}")

        return "\n".join(lines)

    def add_decision(self, decision_text: str):
        self.add("AGENT_DECISION", decision_text)

    def add_scene_event(self, event: dict):
        summary = f"{event.get('event')} {event.get('class')} (track {event.get('track_id')})"
        self.add("SCENE_EVENT", summary)

    def add_tool_result(self, tool_name: str, result: str):
        self.add("TOOL_RESULT", f"{tool_name}: {result[:300]}")
