import argparse
from loguru import logger

from src.utils.config import load_config
from src.pipeline.orchestrator import Orchestrator
from src.pipeline.runner import run_loop

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/dev.yaml")
    p.add_argument("--source", type=str, default=None, help="0 for webcam, or path/rtsp url")
    p.add_argument("--no-window", action="store_true", help="Disable cv2.imshow (headless)")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = load_config(args.config)

    src = args.source if args.source is not None else cfg.get("source", "0")
    
    # normalize source
    if isinstance(src, str) and src.isdigit():
        src = int(src)

    logger.info(f"Config: {args.config}")
    logger.info(f"Source: {src}")

    orch = Orchestrator(cfg)
    run_loop(cfg, src, orch, show_window=not args.no_window)

if __name__ == "__main__":
    main()
