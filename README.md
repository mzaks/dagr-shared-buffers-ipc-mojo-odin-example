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
moves. A forest of 8 trees at depth 14 is **131,064 line segments**, recomputed every
single frame.

The producer also places the trees for depth: each is given a distance, so farther
ones are smaller, sit higher, and are packed back-to-front in the array. The renderer
draws them in that order (near trees occlude far ones) and fades each tree toward a
blue-grey haze by its distance — atmospheric perspective, with no extra data in the
buffer (it re-derives a segment's tree from its index, since `TREES`/`TREE_DEPTH` are
shared schema constants).

## Run it

```bash
pixi install        # fetch the Mojo/MAX toolchain (first time only)
pixi run demo       # or: ./run_demo.sh
```

Close the render window (or press **ESC**) to end the demo; the producer is stopped
automatically.

**GPU or CPU.** The producer computes the forest on the **GPU** when a device is
present and falls back to the **CPU** otherwise — the exact same per-branch function
runs on a GPU thread or in a host loop. You can also force a backend:

```bash
pixi run demo cpu     # force CPU     (or: ./run_demo.sh cpu)
pixi run demo gpu     # force GPU     (or: ./run_demo.sh gpu)
```

(CPU and GPU output match to sub-pixel precision — max ~6e-5 px over the whole
forest.)

At 131,064 segments the two backends pull apart (measured per-frame compute on an
M4 Max, sleep removed):

| Backend | per frame | throughput |
|---|---|---|
| **CPU** (single-threaded loop) | ~8.6 ms | ~116 fps |
| **GPU** (kernel + copy back to the slot) | ~0.44 ms | ~2,280 fps |

So the GPU is **~19×** faster here — and both still clear the demo's 60 fps. The
device→host copy is ~0.2 ms of the GPU's frame; at this scale it's a rounding error
next to the compute the GPU saves. (At the original 12k segments the gap was only
~1.9× — the bigger the forest, the more the parallel GPU pulls ahead.)

**Requirements**

- The **Mojo / MAX toolchain**, via [pixi](https://pixi.sh) — see [`pixi.toml`](pixi.toml).
- An **Apple Silicon GPU** for the GPU path (Mojo's Metal stack); optional — the
  producer runs on the CPU without one.
- The **[Odin](https://odin-lang.org) compiler**, which bundles `vendor:raylib` (no
  separate raylib install needed).

## How the pieces fit

```
  forest_schema.py                     ← one schema, the source of truth
        │  dagr build  (see note below)
        ├────────────► ForestSharedBuffer.mojo   ← Mojo overlay (producer)
        └────────────► forest_sb/                ← Odin overlay package (renderer)

  producer.mojo  ──GPU or CPU──►  branches ──►  ┌─────────────────────┐
                                                │  mmap'd region      │
                        commit() atomic ───────►│  [ctrl][slot0..2]   │
                                                └─────────┬───────────┘
  renderer.odin  ◄──raylib──  latest frame  ◄──consumer_read()──┘
```

- [`producer.mojo`](producer.mojo) — `mmap`s the region and each frame fills an
  off-screen slot with the fractal forest (on the **GPU** via a one-thread-per-branch
  kernel, or the **CPU** via the same function in a host loop) and `commit()`s it
  (one atomic exchange). Runs until stopped.
- [`renderer.odin`](renderer.odin) — `mmap`s the same region, and each frame latches
  the newest published slot and draws every branch with raylib (colour + thickness
  by depth). Trunks and major limbs are thick anti-aliased quads; the ~130k fine
  branches are streamed as one batched `rlgl` GL-line pass (chunk-flushed so none are
  dropped) — which renders the whole forest at ~120 fps instead of ~85.
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
- **One source of truth for the parameters.** The forest's numbers (`TREES`,
  `TREE_DEPTH`, `WORLD_W/H`, sway tuning) and the shared-region `PATH` are declared
  once as **SharedBuffer `constants`** in `forest_schema.py`. The Dagr CLI emits them
  as compile-time constants into *both* overlays, so the Mojo producer and the Odin
  renderer read the same values — nothing is hand-copied between the two languages.

## Related

- **[Dagr SharedBuffers — cross-language FFI & GPU example](https://github.com/mzaks/dagr-shared-buffers-ffi-and-gpu-example)**
  — the same fixed-layout idea *within* a single process: a Mojo GPU kernel drives a
  SharedBuffer in place and hands the bytes to Rust over a C-ABI FFI (both directions),
  with a synchronous `none` hand-off instead of this repo's cross-process
  `double_buffer`.
