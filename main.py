import cv2
import time
import argparse
import traceback
from perception.detector import YOLODetector
from perception.scene_state import SceneState
from perception.stream_handler import StreamHandler
from agent.loop import AgentLoop
from utils.visualizer import draw_detections, draw_agent_annotations, draw_hud
from utils.logger import get_logger

logger = get_logger("main")
from dotenv import load_dotenv
load_dotenv()

def run(show_window: bool = True, save_output: str = None, source=None):
    logger.info("Initializing StreamHandler...")
    stream = StreamHandler(source=source)

    logger.info("Initializing YOLODetector...")
    detector = YOLODetector()

    logger.info("Initializing SceneState...")
    scene = SceneState(stale_timeout=3.0)

    logger.info("Initializing AgentLoop...")
    agent = AgentLoop(scene_state=scene)

    logger.info("Starting stream...")
    stream.start()
    logger.info("Pipeline started. Press Q to quit.")

    writer = None
    last_agent_result = None
    frame_count = 0

    try:
        for frame, timestamp in stream:
            frame_count += 1
            logger.info(f"Frame {frame_count} received — shape: {frame.shape}")

            # ── Perception ──────────────────────────────────────────
            frame_result = detector.detect(frame, timestamp)
            logger.info(f"Detections: {len(frame_result.detections)} objects")

            # ── Scene state update ──────────────────────────────────
            scene.update(frame_result)
            logger.info(f"Scene: {scene.get_snapshot()['class_counts']}")

            # ── Agent loop (rate-gated) ─────────────────────────────
            if agent.should_run():
                logger.info(f"Running agent — batch #{agent._batch_id + 1}")
                try:
                    last_agent_result = agent.run()
                    logger.info(f"Agent decision: {last_agent_result.get('decision')}")
                    logger.info(f"Tools called: {[t['tool'] for t in last_agent_result.get('tool_calls', [])]}")
                except Exception as e:
                    logger.error(f"Agent loop error: {e}")
                    traceback.print_exc()

            # ── Visualization ───────────────────────────────────────
            vis = draw_detections(frame, frame_result)
            if last_agent_result:
                vis = draw_agent_annotations(vis, last_agent_result.get("annotations", []))
            vis = draw_hud(vis, scene.get_snapshot(), last_agent_result)

            # ── Save output ─────────────────────────────────────────
            if save_output:
                if writer is None:
                    h, w = vis.shape[:2]
                    writer = cv2.VideoWriter(
                        save_output, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h)
                    )
                writer.write(vis)

            # ── Display ─────────────────────────────────────────────
            if show_window:
                cv2.imshow("PerceptAgent", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("Q pressed — exiting.")
                    break

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — shutting down.")
    except Exception as e:
        logger.error(f"Pipeline crashed: {e}")
        traceback.print_exc()
    finally:
        logger.info(f"Total frames processed: {frame_count}")
        stream.stop()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        logger.info("Pipeline stopped cleanly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PerceptAgent — YOLO + Agentic Loop")
    parser.add_argument("--no-window", action="store_true", help="Run headless (no display)")
    parser.add_argument("--save", type=str, default=None, help="Save output video to path")
    parser.add_argument("--source", type=str, default=None, help="Override video source (0,1,2 or path)")
    args = parser.parse_args()

    source = args.source
    if source is not None:
        try:
            source = int(source)
        except ValueError:
            pass
        logger.info(f"Source overridden to: {source}")

    run(show_window=not args.no_window, save_output=args.save, source=source)
