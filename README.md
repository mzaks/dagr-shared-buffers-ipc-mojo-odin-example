# Dagr SharedBuffers — cross-language, cross-process IPC (Mojo GPU → Odin raylib)

Two programs, written in **two different languages**, running as **two separate OS
processes**, share **one block of memory** and pass live animation frames back and
forth with **zero serialization**:

- **Process A — Mojo, on the GPU** computes an animated fractal forest and
  *publishes* each frame.
- **Process B — Odin, with raylib** reads the newest frame and *renders* it at
  60 fps.

They never send messages or copy a serialized buffer. They agree on a **fixed
memory layout** and take turns through a lock-free handshake — that's the whole
point of a Dagr **`SharedBuffer`**.

![A fractal forest computed on the GPU in Mojo, streamed through one shared buffer, and rendered by a separate Odin + raylib process](fractal-forest.gif)

*Left process (Mojo, GPU) computes the forest and publishes each frame; right process (Odin, raylib) reads and draws it — sharing one `mmap`'d region, no serialization. Recorded straight from the running demo.*

## Why this works: fixed-layout SharedBuffers

Dagr is a binary format for data graphs. A **`SharedBuffer`** is its fixed-layout
flavour: every field sits at a compile-time-known byte offset, so reading or writing
a field is **pure offset arithmetic** — no varint decoding, no allocation, no
deserialization step. Two consequences make this demo possible:

1. **A GPU kernel can fill the buffer in place.** Offsets are constants, so a Mojo
   kernel writes branch coordinates straight into the region through the generated
   overlay.
2. **Two languages/processes agree on the bytes exactly.** The Mojo overlay and the
   Odin overlay are generated from the *same* schema, so whatever one process writes,
   the other reads with no marshalling in between.

The schema — one `SharedBuffer` called `Forest` — lives in
[`forest_schema.py`](forest_schema.py).

## The handshake: `double_buffer`

A renderer must never see a half-written frame, and it must never make the producer
wait. The `Forest` buffer uses Dagr's **`double_buffer`** concurrency strategy: the
region holds a small atomic control word plus **three** frame slots. The producer
fills an off-screen slot and publishes it with a single atomic exchange; the
consumer always latches the newest *completed* slot and never tears or retries. It's
a classic single-producer / single-consumer triple buffer — and here the producer
and consumer are **different languages in different processes**, coordinating purely
through the shared bytes.

## What the GPU computes

A fractal tree is usually built by recursion, but here every branch is computed
**independently and in parallel**: each GPU thread takes one branch index, decodes
it into a root-to-branch path (the bits of the index pick left/right at each level),
and walks that path to a world transform. Add a time-varying wind term that grows
with depth and the whole forest sways coherently — the trunks stay put, the canopy
moves. A forest of 3 trees at depth 12 is **12,285 line segments**, recomputed every
frame on the GPU.

## Run it

```bash
pixi install        # fetch the Mojo/MAX toolchain (first time only)
pixi run demo       # or: ./run_demo.sh
```

Close the render window (or press **ESC**) to end the demo; the producer is stopped
automatically.

**Requirements**

- An **Apple Silicon GPU** (the producer uses Mojo's Metal GPU stack).
- The **Mojo / MAX toolchain**, via [pixi](https://pixi.sh) — see [`pixi.toml`](pixi.toml).
- The **[Odin](https://odin-lang.org) compiler**, which bundles `vendor:raylib` (no
  separate raylib install needed).

## How the pieces fit

```
  forest_schema.py                     ← one schema, the source of truth
        │  dagr build  (see note below)
        ├────────────► ForestSharedBuffer.mojo   ← Mojo overlay (producer)
        └────────────► forest_sb/                ← Odin overlay package (renderer)

  producer.mojo  ──GPU──►  device node ──copy──►  ┌─────────────────────┐
                                                  │  mmap'd region      │
                          commit() atomic ───────►│  [ctrl][slot0..2]   │
                                                  └─────────┬───────────┘
  renderer.odin  ◄──raylib──  latest frame  ◄──consumer_read()──┘
```

- [`producer.mojo`](producer.mojo) — `mmap`s the region, runs the fractal-forest GPU
  kernel each frame, copies the finished node into an off-screen slot, and
  `commit()`s it (one atomic exchange). Runs until stopped.
- [`renderer.odin`](renderer.odin) — `mmap`s the same region, and each frame latches
  the newest published slot and draws every branch with raylib (colour + thickness
  by depth).
- [`forest_sb/`](forest_sb) — the generated Odin overlay package the renderer
  imports.
- [`ForestSharedBuffer.mojo`](ForestSharedBuffer.mojo) — the generated Mojo overlay
  the producer imports.

The two processes find each other through a fixed file path
(`/tmp/dagr_forest_ipc.bin`) that they both `mmap`.

## About the generated overlays (and the Dagr CLI)

The Mojo overlay and the Odin overlay package are **generated** from
[`forest_schema.py`](forest_schema.py) by the **Dagr CLI**:

```bash
dagr build --schema forest_schema.py
```

> **The Dagr CLI is currently in private beta and not publicly available.**
> So that anyone can build and run this demo without it, the generated overlays are
> **committed to this repo**, and the `dagr build` step in `run_demo.sh` is left
> commented out. Nothing else depends on the CLI.
>
> Interested in trying the Dagr CLI? **Open an issue on this repo** and I'll get in
> touch.

## Notes & scope

- **One producer, one consumer, synchronous-ish.** `double_buffer` is a single-writer
  / single-reader strategy; the atomic control word is the only coordination. Neither
  side ever blocks the other.
- **The GPU-friendly slice of Dagr.** The buffer stores 64-byte-aligned numeric arrays
  (branch endpoints as `f32`, depth as `u8`). Bit-packed and pointer/reference fields
  are deliberately out of scope on-device.
- **Coordinates are in a fixed 1280×800 space** that the renderer draws 1:1; the
  producer and renderer keep that (and the forest depth) in sync as plain constants.
