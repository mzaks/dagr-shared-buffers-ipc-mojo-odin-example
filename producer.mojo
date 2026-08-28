# Fractal-forest GPU producer (process A of the IPC demo).
#
# Each frame, a GPU kernel computes the branch geometry of an animated fractal
# forest — every thread reconstructs ONE branch's world transform from its index
# (no recursion), with a time-varying wind sway — and writes it THROUGH the
# generated Dagr SharedBuffer overlay. The finished node is published into a shared
# mmap'd region with the `double_buffer` strategy's single atomic exchange
# (`producer.commit()`). A separate Odin process (the renderer) latches the newest
# frame and draws it. The two processes never marshal a byte: they share one region
# and one fixed layout.
#
# On an NVIDIA GPU whose driver can reach pageable host memory, there is no device
# node and no copy at all: the kernel dereferences the mmap'd region directly and
# writes its result into the slot the renderer will read.
#
#   Mojo (GPU) --compute--> mmap slot --commit(atomic)-->
#                                             |
#           Odin renderer <--consumer_read()--+
#
# Where that is not available (Metal, or a CUDA device without pageable access), the
# producer falls back to computing into a device node and copying it into the slot:
#
#   Mojo (GPU) --compute--> device node --copy--> mmap slot --commit(atomic)-->

from std.ffi import external_call
from std.gpu import global_idx
from std.math import sin, cos
from std.memory import unsafe_memcpy
from std.sys import argv
from max.gpu.host import DeviceAttribute, DeviceContext, HostBuffer
# TREES / TREE_DEPTH / WORLD_* / L0 / RATIO / THETA / WIND / SPEED / PATH all come
# from the SharedBuffer schema — one source of truth, emitted into this overlay AND
# the Odin renderer's. (They're untyped compile-time constants that adapt to their
# use site, e.g. Float32 in the kernel below.)
from ForestSharedBuffer import (
    Forest, Producer, NODE_SIZE, BYTE_SIZE,
    TREES, TREE_DEPTH, WORLD_W, WORLD_H, L0, RATIO, THETA, WIND, SPEED, PATH,
)

# Derived from the schema constants.
comptime BRANCHES = (1 << TREE_DEPTH) - 1        # 4095 branches per tree
comptime TOTAL    = TREES * BRANCHES             # 12285 segments (<= SEG_CAP)
comptime HALF_PI  = 1.5707963                    # pointing straight up

# Portable POSIX mmap wrapper (macOS + Linux). `PTR` is the raw overlay pointer
# type; `map_shared_file` create-or-opens the region and maps it MAP_SHARED.
from posix_mmap import PTR, map_shared_file


# ── one branch, computed from its index (shared by the GPU and CPU paths) ─────
# Each branch is independent: reconstruct the transform for branch `g` from the bits
# of its heap index (no recursion), apply a time-varying wind sway, and write the
# segment through the SB overlay. The exact same function runs on a GPU thread or in
# a host loop — the overlay's element-region accessors are pure pointer arithmetic.
@always_inline
def write_branch(f: Forest, g: Int, time: Float32):
    var tree = g // BRANCHES
    var b = (g % BRANCHES) + 1                     # 1-based heap index within the tree
    # depth d = floor(log2(b)); trunk (b=1) -> 0
    var d = 0
    var bb = b
    while bb > 1:
        bb >>= 1
        d += 1

    # Perspective layout: tree 0 is the FARTHEST, tree TREES-1 the NEAREST, so the
    # segment array runs back-to-front and the renderer's painter order occludes far
    # trees behind near ones for free. Farther trees are smaller and sit higher up
    # (closer to the horizon); x is scattered with a golden-ratio low-discrepancy
    # sequence so they don't line up in a row.
    var dist: Float32 = 1.0 - Float32(tree) / Float32(TREES - 1)   # 1 = far … 0 = near
    var scale: Float32 = 1.0 - 0.55 * dist                        # far trees ~0.45x size
    var v: Float32 = (Float32(tree) + 0.5) * 0.6180339887
    var xfrac: Float32 = v - Float32(Int(v))                      # fractional part in [0,1)
    var margin: Float32 = 90.0
    var x: Float32 = margin + xfrac * (WORLD_W - 2.0 * margin)    # scattered trunk base X
    var y: Float32 = (WORLD_H - 20.0) - dist * (WORLD_H * 0.42)   # far trees higher on screen
    var ang: Float32 = -HALF_PI                                   # pointing up (screen Y grows downward)
    var ln: Float32 = L0 * scale

    # Walk root -> b, following the bits of b below its MSB (0 = left, 1 = right).
    # Every branch through a given (level, tree) applies the SAME sway, so a child's
    # start point equals its parent's end point exactly — the forest stays connected.
    var k = d - 1
    while k >= 0:
        x += cos(ang) * ln
        y += sin(ang) * ln
        var bit = (b >> k) & 1
        var level = Float32(d - k)
        var sway = WIND * sin(time * SPEED + Float32(tree) * 1.7 + level * 0.6)
        if bit == 1:
            ang += THETA + sway
        else:
            ang += -THETA + sway
        ln *= RATIO
        k -= 1

    # Store through the overlay's `*_ptr()` accessors rather than
    # `set_*_unchecked()`. Both land on the same bytes, but the `set_*` accessors
    # are generated with `alignment=1` (they must serve any field, packed or not),
    # while the element regions are 64-byte aligned — so a 4-byte-aligned store is
    # always valid here. In VRAM the difference is noise (263 vs 272 GB/s), but a
    # kernel writing straight into HOST memory pays for it enormously: an
    # `alignment=1` store lowers to per-byte stores, and each one becomes its own
    # PCIe transaction. Measured on an RTX 4050 over a 2.2 MB node:
    # 0.03 GB/s with alignment=1 vs 12.9 GB/s naturally aligned — a 430x cliff.
    # That cliff is the whole reason the zero-copy path below is viable.
    f.x0_ptr().unsafe_offset(g).unsafe_store(x)
    f.y0_ptr().unsafe_offset(g).unsafe_store(y)
    f.x1_ptr().unsafe_offset(g).unsafe_store(x + cos(ang) * ln)
    f.y1_ptr().unsafe_offset(g).unsafe_store(y + sin(ang) * ln)
    f.depth_ptr().unsafe_offset(g).unsafe_store(UInt8(d))


@always_inline
def set_headers(f: Forest) raises:
    # count + array lengths — constant every frame; the geometry is what changes.
    f.set_count(UInt32(TOTAL))
    f.set_x0_len(UInt32(TOTAL))
    f.set_y0_len(UInt32(TOTAL))
    f.set_x1_len(UInt32(TOTAL))
    f.set_y1_len(UInt32(TOTAL))
    f.set_depth_len(UInt32(TOTAL))


# ── GPU kernels: one thread per branch ────────────────────────────────────────
def k_forest(base: PTR, time: Float32):
    var g = Int(global_idx.x)
    if g < TOTAL:
        write_branch(Forest(base, 0), g, time)     # bare device node (base offset 0)


def k_forest_at(base: PTR, slot_base: Int32, time: Float32):
    # Same kernel, but `base` is the whole shared REGION and `slot_base` is the byte
    # offset of the slot being filled — so the overlay's offset arithmetic lands
    # directly in the mmap'd triple buffer. Used by the zero-copy backend.
    var g = Int(global_idx.x)
    if g < TOTAL:
        write_branch(Forest(base, Int(slot_base)), g, time)


# ── can this device read/write our mmap'd region directly? ───────────────────
# CUDA exposes this as CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS (88): the GPU can
# dereference ordinary host virtual addresses — including a file-backed MAP_SHARED
# mapping — with no cudaHostRegister and no copy. MAX's `DeviceAttribute` is a thin
# newtype over the CUDA enum (there is no named constant for this one yet), and
# `get_attribute` is only meaningful on the CUDA backend, hence the `api()` guard.
comptime CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88

def gpu_can_write_host_memory(ctx: DeviceContext) -> Bool:
    try:
        if ctx.api() != "cuda":
            return False
        return ctx.get_attribute(
            DeviceAttribute(CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS)) == 1
    except:
        return False


# ── backends: fill an off-screen slot each frame, then atomically publish ─────

def run_gpu_zero_copy(ctx: DeviceContext, mut producer: Producer, region: PTR,
                      frames: Int) raises:
    # No device buffer, no H2D, no D2H, no memcpy: the kernel writes the branch
    # geometry straight into the off-screen slot of the shared mmap'd region, which
    # is the very memory the Odin renderer has mapped. `HostBuffer(..., owning=False)`
    # is how a DeviceContext takes an existing host allocation without claiming it —
    # it hands the region to the device side while `unmap` stays our job.
    var shared = HostBuffer[DType.uint8](ctx, region, BYTE_SIZE, owning=False)
    var base = PTR(unsafe_from_address=Int(shared.unsafe_ptr()))
    var grid = (TOTAL + 255) // 256
    var frame = 0
    while frames <= 0 or frame < frames:
        var time = Float32(frame) / 60.0
        var w = producer.write_slot()                         # off-screen slot
        # Headers are 6 scalars — write them on the host, into the slot the kernel is
        # about to fill. (The copy backend can set them once on its device node; here
        # each of the three slots is written in turn, so they are refreshed per frame.)
        set_headers(w)
        w.set_frame(UInt32(frame))
        ctx.enqueue_function[k_forest_at](base, Int32(w.base), time,
                                          grid_dim=grid, block_dim=256)
        ctx.synchronize()                                     # kernel writes now visible
        producer.commit()                                     # atomic publish
        _ = external_call["usleep", Int32](UInt32(16000))     # ~60 fps
        frame += 1
    _ = shared^


def run_gpu(ctx: DeviceContext, mut producer: Producer, frames: Int) raises:
    # Fallback for devices that cannot reach host memory directly: compute the forest
    # on the GPU into a device node, then copy it into the shared slot and commit.
    # Headers are set once on the device node (the kernel only rewrites geometry), so
    # they persist across frames.
    var dev = ctx.enqueue_create_buffer[DType.uint8](NODE_SIZE)
    dev.enqueue_fill(0)
    with dev.map_to_host() as h:
        set_headers(Forest(h.unsafe_ptr(), 0))
    var grid = (TOTAL + 255) // 256
    var frame = 0
    while frames <= 0 or frame < frames:
        var time = Float32(frame) / 60.0
        ctx.enqueue_function[k_forest](dev.unsafe_ptr(), time,
                                       grid_dim=grid, block_dim=256)
        ctx.synchronize()
        with dev.map_to_host() as h:
            Forest(h.unsafe_ptr(), 0).set_frame(UInt32(frame))
            var w = producer.write_slot()                     # off-screen slot
            unsafe_memcpy(dest=w.ptr.unsafe_offset(w.base), src=h.unsafe_ptr(),
                          count=NODE_SIZE)
        producer.commit()                                     # atomic publish
        _ = external_call["usleep", Int32](UInt32(16000))     # ~60 fps
        frame += 1


def run_cpu(mut producer: Producer, frames: Int) raises:
    # The same math on the host: write each branch straight into the shared slot
    # (no device buffer, no copy), then commit. A plain serial loop — 12k branches
    # per frame is trivial for a CPU at 60 fps.
    var frame = 0
    while frames <= 0 or frame < frames:
        var time = Float32(frame) / 60.0
        var w = producer.write_slot()                         # off-screen slot
        set_headers(w)
        w.set_frame(UInt32(frame))
        for g in range(TOTAL):
            write_branch(w, g, time)
        producer.commit()                                     # atomic publish
        _ = external_call["usleep", Int32](UInt32(16000))     # ~60 fps
        frame += 1


def _arg_frames() -> Int:
    # first integer argument = frame count (0 / absent => run until killed)
    var args = argv()
    for i in range(1, len(args)):
        try:
            return Int(String(args[i]))
        except:
            continue
    return 0


def _arg_mode() -> String:
    # "cpu"/"gpu" (or --cpu/--gpu) force the backend; otherwise auto-detect.
    # "gpu-copy" forces the copying GPU backend even where zero-copy is available,
    # which is what the two GPU rows in the README were measured with.
    var args = argv()
    for i in range(1, len(args)):
        var a = String(args[i])
        if a == "cpu" or a == "--cpu":
            return String("cpu")
        if a == "gpu-copy" or a == "--gpu-copy":
            return String("gpu-copy")
        if a == "gpu" or a == "--gpu":
            return String("gpu")
    return String("auto")


def main() raises:
    var mode = _arg_mode()
    var frames = _arg_frames()
    var region = map_shared_file(String(PATH), BYTE_SIZE)
    var producer = Producer.from_raw(region)
    print("producer: forest of", TREES, "trees, depth", TREE_DEPTH, "=", TOTAL,
          "segments; region", BYTE_SIZE, "bytes at", PATH)

    # Auto: use the GPU only if a device is actually present. Explicit cpu/gpu wins.
    var use_gpu = mode == "gpu" or mode == "gpu-copy"
    if mode == "auto":
        use_gpu = DeviceContext.number_of_devices() > 0

    if use_gpu:
        var ctx = DeviceContext()
        # Prefer the zero-copy backend wherever the device can reach host memory.
        if mode != "gpu-copy" and gpu_can_write_host_memory(ctx):
            print("producer: backend = GPU zero-copy —", ctx.name(),
                  "writes the shared region directly (no H2D/D2H copy)")
            run_gpu_zero_copy(ctx, producer, region, frames)
        else:
            if mode == "gpu-copy":
                print("producer: backend = GPU + copy (requested) —", ctx.name())
            else:
                print("producer: backend = GPU + copy —", ctx.name(),
                      "cannot address host memory directly")
            run_gpu(ctx, producer, frames)
    else:
        if mode == "cpu":
            print("producer: backend = CPU (requested)")
        else:
            print("producer: backend = CPU (no GPU device found)")
        run_cpu(producer, frames)
    print("producer: done")
