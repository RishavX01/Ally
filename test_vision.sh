#!/bin/bash
IMG="/tmp/ally/tile_0.png"
B64=$(base64 -w0 "$IMG")

curl -s http://localhost:11434/api/generate -d "{
  \"model\": \"qwen3-vl:4b\",
  \"prompt\": \"What text do you see in this image? Answer briefly.\",
  \"images\": [\"$B64\"],
  \"stream\": false
}" | python3 -c "import sys, json; print(json.load(sys.stdin)['response'])"
