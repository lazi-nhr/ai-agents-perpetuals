from __future__ import annotations

import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    uvicorn.run("defi_ai.services.webhook_server:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
