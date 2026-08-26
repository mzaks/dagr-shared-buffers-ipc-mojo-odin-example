#!/usr/bin/env bash
# Cross-process, cross-language Dagr SharedBuffer demo:
#   process A (Mojo, GPU) computes an animated fractal forest and PUBLISHES frames
#   process B (Odin, raylib) reads the newest frame and RENDERS it at 60 fps
# Both attach to ONE mmap'd region and share a fixed layout — zero marshalling.
#
# Needs: an Apple Silicon GPU, the Mojo/MAX toolchain (via pixi — see pixi.toml),
# and the Odin compiler (with its bundled vendor:raylib). Close the render window
# (or press ESC) to end the demo; the producer is stopped automatically.
set -euo pipefail
cd "$(dirname "$0")"                         # repo root
REGION="/tmp/dagr_forest_ipc.bin"

command -v odin >/dev/null || { echo "error: 'odin' not found — install the Odin compiler"; exit 1; }
command -v pixi >/dev/null || { echo "error: 'pixi' not found — install pixi (https://pixi.sh)"; exit 1; }

# ── [gen] regenerate the SharedBuffer overlays ───────────────────────────────
# The Mojo overlay (ForestSharedBuffer.mojo) and the Odin overlay package
# (forest_sb/) in this repo were produced by the Dagr CLI:
#
#     dagr build --schema forest_schema.py
#
# The CLI is in private beta and not public yet, so the call is commented out and
# the generated overlays are COMMITTED — you can run everything below without it.
# If you have access to the CLI, uncomment to regenerate:
#
# dagr build --schema forest_schema.py --receipt dagr.lock.json

echo "== build renderer (Odin + raylib) =="
odin build . -out:renderer -o:speed

echo "== build producer (Mojo + GPU) =="
pixi run mojo build producer.mojo -o producer

rm -f "$REGION"

# Start the producer (process A). It creates + mmaps the region and publishes
# frames until killed. Any args are forwarded to the producer, so:
#   ./run_demo.sh          # auto: GPU if present, else CPU
#   ./run_demo.sh cpu      # force the CPU backend
#   ./run_demo.sh gpu      # force the GPU backend
# (via pixi: `pixi run demo cpu`).
echo "== launch producer (process A) =="
pixi run ./producer "$@" &
PRODUCER_PID=$!
# Make sure the producer is stopped when the renderer window closes (or on error).
cleanup() { kill "$PRODUCER_PID" 2>/dev/null || true; pkill -f '/producer( |$)' 2>/dev/null || true; }
trap cleanup EXIT

echo "== launch renderer (process B) — close the window to finish =="
./renderer

echo "== done =="
