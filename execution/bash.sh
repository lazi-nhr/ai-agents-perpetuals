docker run -it \
  --user root \
  --name hummingbot_sim \
  --mount "type=bind,source=$HOME/hummingbot_files/scripts,target=/scripts" \
  --mount "type=bind,source=$HOME/Downloads/defi-ai-main,target=/signals" \
  coinalpha/hummingbot:latest

start --script /scripts/json_hyperliquid.py
