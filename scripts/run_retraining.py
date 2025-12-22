from __future__ import annotations

import argparse
from pathlib import Path

from defi_ai.core.config import get_paths
from defi_ai.orchestration.models import load_active_model
from defi_ai.training.retrain import retrain_from_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--default-model", type=str, default="models/best_model.zip")
    args = parser.parse_args()

    paths = get_paths()
    default_model = Path(args.default_model)
    active = load_active_model(paths.model_pointer_file, default_model)

    checkpoint = Path(args.checkpoint) if args.checkpoint else active.path
    res = retrain_from_checkpoint(checkpoint)
    print(f"New model activated: {res.new_model_path} (version={res.version})")


if __name__ == "__main__":
    main()
