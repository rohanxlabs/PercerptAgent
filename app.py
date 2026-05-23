import streamlit as st
import cv2
import groq
import time
import threading
import numpy as np
import os
from PIL import Image
from collections import deque
from datetime import datetime

from perception.detector import YOLODetector
from perception.scene_state import SceneState
from perception.stream_handler import StreamHandler
from agent.loop import AgentLoop
from utils.visualizer import draw_detections, draw_agent_annotations, draw_hud

from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY= os.getenv("GROQ_API_KEY")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PerceptAgent",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
    background-color: #080c10;
    color: #c8d8e8;
}

.stApp {
    background: #080c10;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #1a2a3a;
}

/* Header */
.perceptagent-header {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.6rem;
    color: #00e5ff;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    border-bottom: 1px solid #00e5ff33;
    padding-bottom: 0.4rem;
    margin-bottom: 0.2rem;
}

.perceptagent-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    color: #3a6080;
    letter-spacing: 0.2em;
    margin-bottom: 1.2rem;
}

/* Metric cards */
.metric-card {
    background: #0d1520;
    border: 1px solid #1a3050;
    border-left: 3px solid #00e5ff;
    border-radius: 4px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
}

.metric-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    color: #3a6080;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}

.metric-value {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.6rem;
    color: #00e5ff;
    line-height: 1.2;
}

/* Alert cards */
.alert-critical {
    background: #1a0a0a;
    border-left: 3px solid #ff3d3d;
    padding: 0.5rem 0.8rem;
    border-radius: 3px;
    margin: 0.3rem 0;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    color: #ff8080;
}

.alert-warning {
    background: #1a130a;
    border-left: 3px solid #ffaa00;
    padding: 0.5rem 0.8rem;
    border-radius: 3px;
    margin: 0.3rem 0;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    color: #ffcc66;
}

.alert-info {
    background: #0a1520;
    border-left: 3px solid #00e5ff;
    padding: 0.5rem 0.8rem;
    border-radius: 3px;
    margin: 0.3rem 0;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    color: #80ccff;
}

/* Event log */
.event-log {
    background: #080c10;
    border: 1px solid #1a2a3a;
    border-radius: 4px;
    padding: 0.8rem;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    color: #4a8060;
    max-height: 220px;
    overflow-y: auto;
}

.event-enter { color: #00ff88; }
.event-exit  { color: #ff6060; }

/* Agent decision box */
.decision-box {
    background: #0a1a0a;
    border: 1px solid #00ff8833;
    border-left: 3px solid #00ff88;
    border-radius: 4px;
    padding: 0.8rem 1rem;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    color: #80ffb0;
    line-height: 1.6;
    min-height: 60px;
}

/* Object table */
.obj-row {
    display: flex;
    justify-content: space-between;
    padding: 0.3rem 0.5rem;
    border-bottom: 1px solid #1a2a3a;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
}

.obj-row:hover { background: #0d1520; }

/* Status indicator */
.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse 1.5s infinite;
}

.status-active { background: #00ff88; }
.status-idle   { background: #ffaa00; }
.status-off    { background: #ff3d3d; }

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.3; }
}

/* Section headers */
.section-header {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    color: #3a6080;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    border-bottom: 1px solid #1a2a3a;
    padding-bottom: 0.3rem;
    margin: 0.8rem 0 0.5rem 0;
}

/* Streamlit overrides */
.stButton > button {
    background: #0d1520;
    color: #00e5ff;
    border: 1px solid #00e5ff44;
    border-radius: 3px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    padding: 0.4rem 1.2rem;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: #001a2a;
    border-color: #00e5ff;
    color: #00e5ff;
}

div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    color: #3a6080;
    letter-spacing: 0.1em;
}

.stSelectbox > div > div {
    background: #0d1520;
    border-color: #1a3050;
    color: #c8d8e8;
}
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "running": False,
        "frame": None,
        "detections": [],
        "class_counts": {},
        "active_objects": [],
        "recent_events": deque(maxlen=30),
        "alerts": deque(maxlen=10),
        "agent_decision": "Awaiting first analysis...",
        "agent_tools_called": [],
        "total_frames": 0,
        "agent_calls": 0,
        "fps": 0.0,
        "pipeline": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ── Pipeline thread ───────────────────────────────────────────────────────────
class PipelineThread(threading.Thread):
    def __init__(self, source, conf_threshold):
        super().__init__(daemon=True)
        self.source = source
        self.conf_threshold = conf_threshold
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        import yaml

        # Patch configs at runtime
        with open("configs/yolo_config.yaml") as f:
            ycfg = yaml.safe_load(f)
        try:
            ycfg["stream"]["source"] = int(self.source)
        except (ValueError, TypeError):
            ycfg["stream"]["source"] = self.source
        ycfg["model"]["conf_threshold"] = self.conf_threshold
        with open("configs/yolo_config.yaml", "w") as f:
            yaml.dump(ycfg, f)

        stream = StreamHandler()
        detector = YOLODetector()
        scene = SceneState(stale_timeout=3.0)
        agent = AgentLoop(scene_state=scene)

        stream.start()
        fps_times = deque(maxlen=30)

        try:
            for frame, timestamp in stream:
                if self._stop.is_set():
                    break

                t0 = time.time()

                # Perception
                frame_result = detector.detect(frame, timestamp)
                scene.update(frame_result)
                snapshot = scene.get_snapshot()

                # Agent
                if agent.should_run():
                    try:
                        result = agent.run()
                        st.session_state.agent_decision = result.get("decision", "—")
                        st.session_state.agent_tools_called = [
                            t["tool"] for t in result.get("tool_calls", [])
                        ]
                        st.session_state.agent_calls += 1
                        for alert in result.get("alerts", []):
                            st.session_state.alerts.appendleft(alert)
                    except Exception as e:
                        st.session_state.agent_decision = f"Error: {e}"

                # Visualize
                vis = draw_detections(frame, frame_result)
                vis = draw_hud(vis, snapshot, None)
                vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

                # FPS
                fps_times.append(time.time() - t0)
                fps = 1.0 / (sum(fps_times) / len(fps_times)) if fps_times else 0

                # Update session state
                st.session_state.frame = vis_rgb
                st.session_state.detections = [d.to_dict() for d in frame_result.detections]
                st.session_state.class_counts = snapshot["class_counts"]
                st.session_state.active_objects = snapshot["active_objects"]
                st.session_state.total_frames += 1
                st.session_state.fps = round(fps, 1)

                for evt in snapshot.get("recent_events", []):
                    if evt not in list(st.session_state.recent_events):
                        st.session_state.recent_events.appendleft(evt)

        finally:
            stream.stop()
            st.session_state.running = False


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="perceptagent-header">👁 PerceptAgent</div>', unsafe_allow_html=True)
    st.markdown('<div class="perceptagent-sub">YOLO · GROQ · AGENTIC LOOP</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">// Source</div>', unsafe_allow_html=True)
    source_type = st.selectbox("Input", ["Webcam", "Video File"], label_visibility="collapsed")

    if source_type == "Webcam":
        cam_index = st.selectbox("Camera Index", [0, 1, 2, 3])
        source = cam_index
    else:
        video_path = st.text_input("Video path", placeholder="path/to/video.mp4")
        source = video_path

    st.markdown('<div class="section-header">// Detection</div>', unsafe_allow_html=True)
    conf_thresh = st.slider("Confidence", 0.1, 0.9, 0.35, 0.05)

    st.markdown('<div class="section-header">// Control</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button("▶ START", use_container_width=True)
    with col2:
        stop_btn = st.button("■ STOP", use_container_width=True)

    if start_btn and not st.session_state.running:
        st.session_state.running = True
        pipeline = PipelineThread(source=source, conf_threshold=conf_thresh)
        pipeline.start()
        st.session_state.pipeline = pipeline

    if stop_btn and st.session_state.running:
        if st.session_state.pipeline:
            st.session_state.pipeline.stop()
        st.session_state.running = False

    # Status
    st.markdown('<div class="section-header">// Status</div>', unsafe_allow_html=True)
    if st.session_state.running:
        st.markdown('<span class="status-dot status-active"></span> **LIVE**', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-dot status-off"></span> **OFFLINE**', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Frames Processed</div>
        <div class="metric-value">{st.session_state.total_frames}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">FPS</div>
        <div class="metric-value">{st.session_state.fps}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Agent Calls</div>
        <div class="metric-value">{st.session_state.agent_calls}</div>
    </div>
    """, unsafe_allow_html=True)


# ── Main layout ───────────────────────────────────────────────────────────────
col_feed, col_right = st.columns([3, 2], gap="medium")

with col_feed:
    st.markdown('<div class="section-header">// Live Feed</div>', unsafe_allow_html=True)
    feed_placeholder = st.empty()

    if st.session_state.frame is not None:
        feed_placeholder.image(st.session_state.frame, use_container_width=True)
    else:
        feed_placeholder.markdown("""
        <div style="
            background:#0d1520;
            border:1px dashed #1a3050;
            border-radius:4px;
            height:380px;
            display:flex;
            align-items:center;
            justify-content:center;
            font-family:'Share Tech Mono',monospace;
            font-size:0.8rem;
            color:#1a3050;
            letter-spacing:0.2em;
        ">NO SIGNAL — PRESS START</div>
        """, unsafe_allow_html=True)

    # Agent decision
    st.markdown('<div class="section-header">// Agent Decision</div>', unsafe_allow_html=True)
    tools_str = " · ".join(st.session_state.agent_tools_called) if st.session_state.agent_tools_called else "none"
    st.markdown(f"""
    <div class="decision-box">
        {st.session_state.agent_decision}<br><br>
        <span style="color:#2a6040;font-size:0.65rem;">TOOLS: {tools_str}</span>
    </div>
    """, unsafe_allow_html=True)


with col_right:
    # Object counts
    st.markdown('<div class="section-header">// Detected Objects</div>', unsafe_allow_html=True)
    if st.session_state.class_counts:
        for cls, count in st.session_state.class_counts.items():
            bar_pct = min(count * 20, 100)
            st.markdown(f"""
            <div class="obj-row">
                <span style="color:#c8d8e8">{cls}</span>
                <span style="color:#00e5ff">{count}</span>
            </div>
            <div style="background:#1a2a3a;border-radius:2px;height:3px;margin-bottom:4px;">
                <div style="background:#00e5ff;width:{bar_pct}%;height:3px;border-radius:2px;"></div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown('<div style="font-family:\'Share Tech Mono\',monospace;font-size:0.7rem;color:#1a3050;padding:0.5rem;">No objects detected</div>', unsafe_allow_html=True)

    # Alerts
    st.markdown('<div class="section-header">// Alerts</div>', unsafe_allow_html=True)
    if st.session_state.alerts:
        for alert in list(st.session_state.alerts)[:5]:
            atype = alert.get("type", "info")
            desc = alert.get("description", "")
            conf = alert.get("confidence", 0)
            css_class = "alert-critical" if conf > 0.8 else "alert-warning"
            st.markdown(f"""
            <div class="{css_class}">
                [{atype.upper()}] {desc}<br>
                <span style="opacity:0.5">conf: {conf:.2f}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert-info">System nominal — no alerts</div>', unsafe_allow_html=True)

    # Event log
    st.markdown('<div class="section-header">// Event Log</div>', unsafe_allow_html=True)
    events_html = '<div class="event-log">'
    if st.session_state.recent_events:
        for evt in list(st.session_state.recent_events)[:15]:
            ts = datetime.fromtimestamp(evt.get("timestamp", time.time())).strftime("%H:%M:%S")
            etype = evt.get("event", "")
            cls = evt.get("class", "")
            tid = evt.get("track_id", "")
            css = "event-enter" if etype == "ENTER" else "event-exit"
            events_html += f'<div><span style="color:#2a4060">{ts}</span> <span class="{css}">{etype}</span> <span style="color:#c8d8e8">{cls}</span> <span style="color:#2a4060">#{tid}</span></div>'
    else:
        events_html += '<div style="color:#1a3050">Waiting for events...</div>'
    events_html += '</div>'
    st.markdown(events_html, unsafe_allow_html=True)

    # Active tracked objects detail
    st.markdown('<div class="section-header">// Tracked Objects</div>', unsafe_allow_html=True)
    if st.session_state.active_objects:
        for obj in st.session_state.active_objects[:8]:
            dur = obj.get("duration_sec", 0)
            tid = obj.get("track_id", "?")
            cls = obj.get("class_name", "?")
            conf = obj.get("confidence_avg", 0)
            st.markdown(f"""
            <div class="obj-row">
                <span style="color:#3a6080">#{tid}</span>
                <span style="color:#c8d8e8">{cls}</span>
                <span style="color:#00e5ff">{dur:.1f}s</span>
                <span style="color:#2a6040">{conf:.2f}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown('<div style="font-family:\'Share Tech Mono\',monospace;font-size:0.7rem;color:#1a3050;padding:0.5rem;">No tracked objects</div>', unsafe_allow_html=True)


# ── Auto-refresh ──────────────────────────────────────────────────────────────
if st.session_state.running:
    time.sleep(0.1)
    st.rerun()
