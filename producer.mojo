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
#   Mojo (GPU) --compute--> device node --copy--> mmap slot --commit(atomic)-->
#                                                                    |
#                                          Odin renderer <--consumer_read()------+

from std.ffi import external_call
from std.gpu import global_idx
from std.math import sin, cos
from std.memory import unsafe_memcpy
from std.sys import argv
from max.gpu.host import DeviceContext
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

# macOS open/mmap constants
comptime O_RDWR = 0x0002
comptime O_CREAT = 0x0200
comptime PROT_RW = 0x3
comptime MAP_SHARED = 0x1
comptime PTR = Pointer[UInt8, MutUntrackedOrigin]


def map_shared(path: String, size: Int) -> PTR:
    var fd = external_call["open", Int32](
        path.unsafe_ptr(), Int32(O_RDWR | O_CREAT), Int32(0o666))
    _ = external_call["fchmod", Int32](fd, Int32(0o666))       # let the renderer (same user) open it
    _ = external_call["ftruncate", Int32](fd, Int64(size))
    var addr = external_call["mmap", PTR](
        Int(0), size, Int32(PROT_RW), Int32(MAP_SHARED), fd, Int64(0))
    _ = external_call["close", Int32](fd)
    return addr


# ── one branch, computed from its index (shared by the GPU and CPU paths) ─────
# Each branch is independent: reconstruct the transform for branch `g` from the bits
# of its heap index (no recursion), apply a time-varying wind sway, and write the
# segment through the SB overlay. The exact same function runs on a GPU thread or in
# a host loop — the overlay's `*_unchecked` accessors are pure pointer arithmetic.
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

    var spacing = Float32(WORLD_W) / Float32(TREES)
    var x: Float32 = spacing * (Float32(tree) + 0.5)   # trunk base X
    var y: Float32 = WORLD_H - 12.0                     # trunk base Y (near the ground)
    var ang: Float32 = -HALF_PI                         # pointing up (screen Y grows downward)
    var ln: Float32 = L0

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

    f.set_x0_unchecked(g, x)
    f.set_y0_unchecked(g, y)
    f.set_x1_unchecked(g, x + cos(ang) * ln)
    f.set_y1_unchecked(g, y + sin(ang) * ln)
    f.set_depth_unchecked(g, UInt8(d))


@always_inline
def set_headers(f: Forest) raises:
    # count + array lengths — constant every frame; the geometry is what changes.
    f.set_count(UInt32(TOTAL))
    f.set_x0_len(UInt16(TOTAL))
    f.set_y0_len(UInt16(TOTAL))
    f.set_x1_len(UInt16(TOTAL))
    f.set_y1_len(UInt16(TOTAL))
    f.set_depth_len(UInt16(TOTAL))


# ── GPU kernel: one thread per branch ─────────────────────────────────────────
def k_forest(base: PTR, time: Float32):
    var g = Int(global_idx.x)
    if g < TOTAL:
        write_branch(Forest(base, 0), g, time)     # bare device node (base offset 0)


# ── backends: fill an off-screen slot each frame, then atomically publish ─────

def run_gpu(mut producer: Producer, frames: Int) raises:
    # Compute the forest on the GPU into a device node, then copy it into the shared
    # slot and commit. Headers are set once on the device node (the kernel only
    # rewrites geometry), so they persist across frames.
    var ctx = DeviceContext()
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
    var args = argv()
    for i in range(1, len(args)):
        var a = String(args[i])
        if a == "cpu" or a == "--cpu":
            return String("cpu")
        if a == "gpu" or a == "--gpu":
            return String("gpu")
    return String("auto")


def main() raises:
    var mode = _arg_mode()
    var frames = _arg_frames()
    var region = map_shared(String(PATH), BYTE_SIZE)
    var producer = Producer.from_raw(region)
    print("producer: forest of", TREES, "trees, depth", TREE_DEPTH, "=", TOTAL,
          "segments; region", BYTE_SIZE, "bytes at", PATH)

    # Auto: use the GPU only if a device is actually present. Explicit cpu/gpu wins.
    var use_gpu = mode == "gpu"
    if mode == "auto":
        use_gpu = DeviceContext.number_of_devices() > 0

    if use_gpu:
        print("producer: backend = GPU")
        run_gpu(producer, frames)
    else:
        if mode == "cpu":
            print("producer: backend = CPU (requested)")
        else:
            print("producer: backend = CPU (no GPU device found)")
        run_cpu(producer, frames)
    print("producer: done")
