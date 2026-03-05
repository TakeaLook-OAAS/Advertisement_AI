from loguru import logger
from src.pipeline.orchestrator import Orchestrator
from src.pipeline.runner import run_loop
from src.utils.config import load_config

def main():
    cfg = load_config("configs/dev.yaml")
    src = cfg.get("source", "0")  # yaml → 없으면 웹캠 "0"

    # normalize source
    if isinstance(src, str) and src.isdigit():
        src = int(src)

    logger.info(f"Source: {src}")

    orch = Orchestrator(cfg)
    run_loop(cfg, src, orch)

if __name__ == "__main__":
    main()
