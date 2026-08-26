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


# ── GPU kernel: one thread computes one branch, fully in parallel ─────────────
def k_forest(base: PTR, time: Float32):
    var g = Int(global_idx.x)
    if g >= TOTAL:
        return
    var f = Forest(base, 0)                       # bare device node (base offset 0)

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


def _arg_frames() -> Int:
    # optional argv[1] = frame count (0 or absent => run until killed)
    var args = argv()
    if len(args) > 1:
        try:
            return Int(String(args[1]))
        except:
            return 0
    return 0


def main() raises:
    var region = map_shared(String(PATH), BYTE_SIZE)
    var producer = Producer.from_raw(region)

    var ctx = DeviceContext()
    var dev = ctx.enqueue_create_buffer[DType.uint8](NODE_SIZE)
    dev.enqueue_fill(0)
    # Set the count + array lengths once on the device node; the kernel only rewrites
    # the geometry arrays each frame, so these headers persist across frames.
    with dev.map_to_host() as h:
        var f = Forest(h.unsafe_ptr(), 0)
        f.set_count(UInt32(TOTAL))
        f.set_x0_len(UInt16(TOTAL))
        f.set_y0_len(UInt16(TOTAL))
        f.set_x1_len(UInt16(TOTAL))
        f.set_y1_len(UInt16(TOTAL))
        f.set_depth_len(UInt16(TOTAL))

    var frames = _arg_frames()
    var grid = (TOTAL + 255) // 256
    var frame = 0
    print("producer: forest of", TREES, "trees, depth", TREE_DEPTH, "=", TOTAL,
          "segments; region", BYTE_SIZE, "bytes at", PATH)
    while frames <= 0 or frame < frames:
        var time = Float32(frame) / 60.0
        ctx.enqueue_function[k_forest](dev.unsafe_ptr(), time,
                                       grid_dim=grid, block_dim=256)
        ctx.synchronize()
        with dev.map_to_host() as h:
            var f = Forest(h.unsafe_ptr(), 0)
            f.set_frame(UInt32(frame))
            var w = producer.write_slot()                     # off-screen slot
            unsafe_memcpy(dest=w.ptr.unsafe_offset(w.base), src=h.unsafe_ptr(),
                          count=NODE_SIZE)
        producer.commit()                                     # atomic publish
        _ = external_call["usleep", Int32](UInt32(16000))     # ~60 fps
        frame += 1
    print("producer: done after", frame, "frames")
