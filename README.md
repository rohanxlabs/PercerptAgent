# 👁 PerceptAgent

> Real-time YOLO perception + Groq-powered agentic decision loop

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple?style=flat-square)
![Groq](https://img.shields.io/badge/LLM-Groq-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

PerceptAgent detects and tracks objects in live video using YOLOv8, builds a structured scene representation, and feeds it to an LLM agent that reasons about the scene and executes tool-based actions — all inside a production-grade Streamlit dashboard.

---

## Pipeline

```
Camera / Video
      │
      ▼
StreamHandler ──► YOLODetector ──► SceneState
                        │               │
                  Detects &        Tracks objects,
                  tracks objects   logs ENTER/EXIT
                                        │
                                        ▼
                                   AgentLoop (Groq LLM)
                                        │
                    ┌───────────────────┼───────────────────┐
               log_event        trigger_alert          send_command
               annotate_frame   query_object_history
```

---

## Features

- **Real-time detection** — YOLOv8 with ByteTrack multi-object tracking
- **Scene graph** — tracks object lifecycle, dwell time, ENTER/EXIT events
- **Agentic loop** — ReAct-style LLM agent: reason → call tools → observe → repeat
- **5 agent tools** — log events, trigger alerts, query history, annotate frames, send commands
- **Episodic memory** — sliding window of past decisions for agent continuity
- **Streamlit dashboard** — live feed, detections, agent decisions, alert panel, event log
- **Robot-ready** — `send_command` tool hooks into ROS2, serial, or HTTP

---

## Project Structure

```
perceptagent/
├── perception/
│   ├── detector.py          # YOLOv8 wrapper with ByteTrack
│   ├── scene_state.py       # Scene graph & object lifecycle
│   └── stream_handler.py    # Threaded video capture
├── agent/
│   ├── loop.py              # Groq ReAct agent loop
│   ├── tools.py             # Tool schemas + executor
│   ├── prompts.py           # System + context prompts
│   └── memory.py            # Episodic memory
├── configs/
│   ├── agent_config.yaml    # Model, loop, memory params
│   └── yolo_config.yaml     # YOLO model, tracking, stream
├── utils/
│   ├── visualizer.py        # OpenCV annotation helpers
│   └── logger.py            # Structured logger
├── tests/
├── app.py                   # Streamlit dashboard
├── main.py                  # Headless CLI runner
└── requirements.txt
```

---

## Installation

**1. Clone**
```bash
git clone https://github.com/rohanxlabs/perceptagent
cd perceptagent
```

**2. Install dependencies**
```bash
pip install ultralytics groq opencv-python numpy pyyaml streamlit
```

**3. Set Groq API key**
```bash
# Linux / macOS
export GROQ_API_KEY="gsk_..."

# Windows CMD
set GROQ_API_KEY=gsk_...

# Windows PowerShell
$env:GROQ_API_KEY="gsk_..."
```

> Get a free key at [console.groq.com](https://console.groq.com)

---

## Running

**Streamlit dashboard**
```bash
streamlit run app.py
# Opens at http://localhost:8501
```

Use the sidebar to select your video source, set confidence threshold, and click **▶ START**.

**Headless CLI**
```bash
python main.py                        # default webcam
python main.py --source 1             # webcam index 1
python main.py --source video.mp4     # video file
python main.py --save output.mp4      # save annotated output
python main.py --no-window            # headless
```

---

## Configuration

**configs/yolo_config.yaml**
```yaml
model:
  weights: "yolov8n.pt"    # yolov8n/s/m/l/x
  device: "cpu"            # cpu | cuda | mps
  conf_threshold: 0.35
  iou_threshold: 0.45
tracking:
  enabled: true
  tracker: "bytetrack.yaml"
stream:
  source: 0                # 0=webcam or path to video file
  max_fps: 30
```

**configs/agent_config.yaml**
```yaml
model:
  name: "llama-3.3-70b-versatile"
  max_tokens: 1024
  temperature: 0.3
loop:
  max_iterations: 10
  frame_batch_size: 5      # frames between agent calls
  cooldown_frames: 15      # skip frames after action taken
memory:
  max_events: 20
```

---

## Agent Tools

| Tool | Purpose | Key Parameters |
|---|---|---|
| `log_event` | Log a notable scene event | severity, message, objects_involved |
| `trigger_alert` | Fire a high-priority alert | alert_type, description, confidence |
| `query_object_history` | Query dwell time and behavior | class_name |
| `annotate_frame` | Request custom frame annotation | label, color, track_id |
| `send_command` | Send command to external system | target, action, parameters |

To connect `send_command` to your robot arm or ROS2 topic, edit `_tool_send_command` in `agent/tools.py`.

---

## Groq Model Options

| Model | Speed | Best For |
|---|---|---|
| `llama-3.3-70b-versatile` | Fast | Best reasoning — default |
| `llama3-8b-8192` | Fastest | Low-latency, simpler scenes |
| `mixtral-8x7b-32768` | Fast | Large context, complex history |

Change model by editing `model.name` in `configs/agent_config.yaml`.

---

## Customizing Agent Behavior

Edit the system prompt in `agent/prompts.py`:

```python
# Security / surveillance
"- Alert if more than 3 persons detected simultaneously"
"- Alert if a person dwells for more than 10 seconds"

# Robotics / manipulation
"- Send robot_arm command when target object detected"
"- Query object history before commanding a pick action"

# Warehouse / counting
"- Log event when item count changes"
"- Alert if unknown object class appears on conveyor"
```

---

## Connecting to a Robot Arm

Replace the stub in `agent/tools.py`:

```python
def _tool_send_command(self, target, action, parameters=None):
    if target == "robot_arm":
        # Option A: ROS2 topic
        subprocess.run(["ros2", "topic", "pub", "/arm/cmd", ...])

        # Option B: Serial (Feetech STS3215)
        self.serial_port.write(build_servo_packet(action, parameters))

        # Option C: HTTP to FastAPI
        requests.post("http://localhost:8000/command", json={
            "action": action, "params": parameters
        })
```

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

Covers: Detection parsing, FrameResult summary, all 5 ToolExecutor tools, EpisodicMemory overflow and context.

---

## Troubleshooting

| Issue | Fix |
|---|---|
| Exits immediately | Check last log line — run with debug logging in main.py |
| Cannot open source: 0 | Try source: 1 or 2 in yolo_config.yaml, or use a video file |
| ModuleNotFoundError | Ensure `__init__.py` exists in perception/, agent/, utils/ |
| GROQ_API_KEY not set | Export env variable — verify with `echo $GROQ_API_KEY` |
| YOLO download fails | First run downloads weights (~6MB) — needs internet |
| Streamlit blank feed | Click START in sidebar — feed only starts after button press |
| Low FPS | Use yolov8n.pt, lower imgsz to 320, increase frame_batch_size |

---

## Roadmap

- [ ] ROS2 integration — publish detections as sensor_msgs
- [ ] MoveIt2 bridge — agent triggers grasp/place via action servers
- [ ] Zone definition — draw restricted zones in UI
- [ ] Multi-camera support — aggregate scene state across feeds
- [ ] ONNX export — run YOLO on-device (Jetson Nano, Raspberry Pi)
- [ ] Voice alerts — TTS output for agent decisions

---

## Author

Built by **Rohan** · [github.com/rohanxlabs](https://github.com/rohanxlabs) · Pre-CSE AI/ML student building in public

Stack: Python · YOLOv8 · Groq · Streamlit · ROS2 · MuJoCo · Anthropic SDK

---

## License

MIT License — free to use, modify, and distribute. Attribution appreciated.
