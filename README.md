# PerceptAgent

PerceptAgent turns live video into tracked scene context, then lets a Groq-backed agent choose validated tools whose results feed the next decision.

## Problem

Object detection alone does not retain context or act. PerceptAgent provides a small, inspectable perception → memory → decision → tool → observation loop for video monitoring experiments.

## Architecture

```text
camera or video
      │
      ▼
StreamHandler ──► YOLODetector + ByteTrack ──► SceneState
                                                │
                         scene events ─────────┤
                                                ▼
EpisodicMemory ◄── tool results ◄── AgentLoop (Groq, optional)
                                      │
                                      ▼
                         ToolExecutor (validated JSON inputs)
                                      │
                                      ▼
                    SimulatedActuator (explicitly simulation only)
```

## What works

- YOLOv8 detection and ByteTrack IDs via Ultralytics.
- A compact scene state with active tracked objects and ENTER/EXIT events.
- A Groq function-calling loop that returns tool results to the model for another iteration.
- Bounded in-process episodic memory of scene events, decisions, and tool results.
- Five validated tools: `log_event`, `trigger_alert`, `query_object_history`, `annotate_frame`, and `send_command`.
- `send_command` is a safe simulated action adapter. It does not control physical hardware.

## Installation

Python 3.10+ is required.

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows PowerShell/cmd equivalent is fine
pip install -r requirements.txt
copy .env.example .env
```

Set `GROQ_API_KEY` in `.env` only if using the LLM agent. Never commit it. The offline demo below needs no key.

## Run

Use a camera or a video file:

```bash
python main.py --source path\to\video.mp4
python main.py --source 0
streamlit run app.py
```

The CLI/dashboard require a Groq key because they instantiate the LLM agent. If the key is unavailable, run the reproducible offline integration demo instead:

```bash
python scripts/demo.py --source path\to\video.mp4 --frames 60 --output artifacts/demo.mp4
```

It runs real detection/tracking from the supplied video, stores scene events in memory, executes a validated simulated pointing command for a newly tracked person, and prints the action/memory record. The generated annotated video is suitable for the perception screenshot; the printed JSON supplies the memory/action evidence. This is a deterministic policy fallback, not LLM reasoning.

## LLM agent flow

For every rate-gated batch, `AgentLoop` builds a prompt from `SceneState` and episodic memory. The model can request a tool using Groq function calling. Tool arguments are parsed and validated, the JSON result is appended as a tool message, and the agent can make another decision up to `max_iterations`. Invalid JSON, invalid arguments, tool failures, missing keys, model failures, unavailable input, and model-load/inference errors return explicit errors rather than fabricated success.

## Configuration

- `configs/yolo_config.yaml`: model weights, thresholds, tracker, and default source.
- `configs/agent_config.yaml`: model and loop/memory limits.
- CLI and dashboard source/threshold choices are runtime overrides; they do not edit those files.

## Project structure

```text
perception/  detection, tracking state, input stream
agent/       LLM loop, prompts, memory, tools, simulated action adapter
scripts/     reproducible offline demo
tests/       unit and integration-path tests
```

## Limitations

- No physical robot, ROS2 bridge, persistence layer, API/backend, or deployment manifest is implemented.
- Tracking quality and IDs depend on the selected model, tracker, video, and hardware.
- The LLM loop requires a valid Groq account/key and network access.
- The Streamlit UI has not been browser-tested in this repository environment.

## Verification

```bash
pytest tests -v
```

Tests cover detector result objects, scene → memory → tool → simulated action flow, memory bounds, and tool input rejection. A live YOLO inference test requires dependencies plus a supplied video/image and is performed with the demo command above.
