# posix_mmap.mojo — portable POSIX mmap for Mojo (macOS + Linux; both via libc).
#
# Reference implementation for a proposed Mojo stdlib `mmap`. The design follows
# the two mature precedents:
#
#   * Zig  — `std.posix.mmap`: ONE portable wrapper; every OS/arch difference is
#            pushed into typed constant tables (`PROT`/`MAP`/`O` are packed
#            structs, not loose `#define` ints).
#   * Rust — `libc` (per-target constants) + `memmap2` (a `Mmap`/`MmapMut` type
#            that hides page-alignment of the offset and owns the mapping via RAII).
#
# WHY THIS IS A CONSTANTS PROBLEM, NOT AN ABI PROBLEM (for Mojo today):
# Mojo's `CompilationTarget` exposes `is_macos()`/`is_linux()` but NOT
# `is_windows()`/`is_freebsd()` — Mojo targets only macOS and Linux, both POSIX,
# both reached through libc. The libc `mmap` ABI is identical on those targets:
# `off_t` is 64-bit, the offset is a byte offset, and the plain `mmap` symbol is
# used (the `mmap2`/page-scaled-offset variant is only for 32-bit Linux archs; no
# `__mmap` quirk on Darwin). So the `external_call` shape is the same everywhere;
# only a handful of flag VALUES differ, selected below at comptime.
#
# NOT covered (deliberately): Windows (`CreateFileMappingW`/`MapViewOfFile`, offset
# aligned to the 64 KB allocation granularity, split hi/lo dwords). That is a
# separate code path selected at the call site — Zig's std does not unify it under
# one `mmap` signature and neither should Mojo's until a Windows target exists.
#
# Constant values cross-checked against the real macOS and Linux headers.
from std.sys import CompilationTarget
from std.ffi import external_call

comptime PTR = Pointer[UInt8, MutUntrackedOrigin]


# ─────────────────────────────────────────────────────────────────────────────
# Typed flag sets (Zig-style). Newtypes over Int32 with `|` composition, so the
# compiler stops you from passing a `PROT_*` where a `MAP_*` is expected — and the
# per-OS value lives in exactly one place. All fold at comptime.
# ─────────────────────────────────────────────────────────────────────────────
@fieldwise_init
struct Prot(TrivialRegisterPassable, Writable):
    """Memory protection flags for `mmap`/`mprotect` (identical on macOS & Linux)."""
    var value: Int32
    def __or__(self, other: Self) -> Self:
        return Self(self.value | other.value)
    def write_to(self, mut w: Some[Writer]):
        w.write("Prot(", hex(Int(self.value)), ")")

comptime PROT_NONE  = Prot(0x0)
comptime PROT_READ  = Prot(0x1)
comptime PROT_WRITE = Prot(0x2)
comptime PROT_EXEC  = Prot(0x4)


@fieldwise_init
struct MapFlags(TrivialRegisterPassable, Writable):
    """`mmap` flags. MAP_ANONYMOUS differs: macOS 0x1000, Linux/x86_64 0x20."""
    var value: Int32
    def __or__(self, other: Self) -> Self:
        return Self(self.value | other.value)
    def write_to(self, mut w: Some[Writer]):
        w.write("MapFlags(", hex(Int(self.value)), ")")

comptime MAP_SHARED    = MapFlags(0x1)
comptime MAP_PRIVATE   = MapFlags(0x2)
comptime MAP_FIXED     = MapFlags(0x10)
comptime MAP_ANONYMOUS = MapFlags(Int32(0x1000 if CompilationTarget.is_macos() else 0x20))


@fieldwise_init
struct OFlags(TrivialRegisterPassable, Writable):
    """`open` flags. O_CREAT/O_EXCL/O_TRUNC differ between macOS and Linux."""
    var value: Int32
    def __or__(self, other: Self) -> Self:
        return Self(self.value | other.value)
    def write_to(self, mut w: Some[Writer]):
        w.write("OFlags(", hex(Int(self.value)), ")")

comptime O_RDONLY = OFlags(0x0)
comptime O_WRONLY = OFlags(0x1)
comptime O_RDWR   = OFlags(0x2)
comptime O_CREAT  = OFlags(Int32(0x0200 if CompilationTarget.is_macos() else 0x40))
comptime O_EXCL   = OFlags(Int32(0x0800 if CompilationTarget.is_macos() else 0x80))
comptime O_TRUNC  = OFlags(Int32(0x0400 if CompilationTarget.is_macos() else 0x200))


@fieldwise_init
struct SyncFlags(TrivialRegisterPassable, Writable):
    """`msync` flags. MS_SYNC differs: macOS 0x10, Linux 0x4."""
    var value: Int32
    def __or__(self, other: Self) -> Self:
        return Self(self.value | other.value)
    def write_to(self, mut w: Some[Writer]):
        w.write("SyncFlags(", hex(Int(self.value)), ")")

comptime MS_ASYNC      = SyncFlags(0x1)
comptime MS_INVALIDATE = SyncFlags(0x2)
comptime MS_SYNC       = SyncFlags(Int32(0x10 if CompilationTarget.is_macos() else 0x4))

# `sysconf` selector for the page size — ALSO differs: macOS 29, Linux 30.
comptime _SC_PAGESIZE = 29 if CompilationTarget.is_macos() else 30


# ─────────────────────────────────────────────────────────────────────────────
# Low-level libc wrappers (mirror `std.posix`: page-aligned offset required here;
# the `Mapping` type below relaxes that).
# ─────────────────────────────────────────────────────────────────────────────
@always_inline
def page_size() -> Int:
    """Runtime page size via `sysconf(_SC_PAGESIZE)` (16384 on Apple Silicon)."""
    return Int(external_call["sysconf", Int64](Int32(_SC_PAGESIZE)))


@always_inline
def mmap(length: Int, prot: Prot, flags: MapFlags, fd: Int32, offset: Int64 = 0) -> PTR:
    """Plain libc `mmap`, 64-bit byte offset. `offset` must be page-aligned.
    Returns MAP_FAILED ((void*)-1) on error — check with `map_failed`."""
    return external_call["mmap", PTR](Int(0), length, prot.value, flags.value, fd, offset)


@always_inline
def munmap(addr: PTR, length: Int) -> Int32:
    return external_call["munmap", Int32](addr, length)


@always_inline
def msync(addr: PTR, length: Int, flags: SyncFlags = MS_SYNC) -> Int32:
    """Flush a mapped region to backing store (default MS_SYNC = durable)."""
    return external_call["msync", Int32](addr, length, flags.value)


@always_inline
def map_failed(p: PTR) -> Bool:
    """mmap failure sentinel is MAP_FAILED = (void*)-1 (all-ones), NOT null."""
    return Int(p) == -1


# ─────────────────────────────────────────────────────────────────────────────
# `Mapping` — owns a mapping and hides the arbitrary-offset page-alignment shim
# (the `memmap2` model). mmap requires a page-aligned offset, but callers want an
# arbitrary byte offset: we round the offset DOWN to a page boundary, grow the
# length by the same delta, map that, and hand back `base + delta` from `ptr()`.
#
# OWNERSHIP IS EXPLICIT (linear), NOT RAII. A tempting `__deinit__` that calls
# `munmap` is UNSAFE in current Mojo: `ptr()` hands out a pointer with an
# untracked origin, so it does not extend this value's lifetime. Under Mojo's
# ASAP destruction the compiler then frees the mapping (munmap) at the owner's
# last *recognized* use — which can land BEFORE a later read/write through an
# escaped pointer (verified: the store lands in already-unmapped memory). Even a
# tracked-origin `Span[UInt8, origin_of(self)]` accessor did not reliably extend
# the owner's life across the span's later uses in this toolchain. So the region
# is freed only by an explicit `unmap(deinit self)`; dropping a `Mapping` without
# calling it leaks the mapping until process exit (same as a bare `mmap`). This
# is the crux of what a stdlib owning-mmap type needs from the language: an
# accessor whose origin keeps the owner alive for the *escaped pointer's* whole
# lifetime.  `Mapping` is `Movable`-only so it cannot be silently duplicated.
# ─────────────────────────────────────────────────────────────────────────────
struct Mapping(Movable):
    var _base: PTR          # page-aligned base returned by mmap (what munmap needs)
    var _ptr: PTR           # user pointer = base + alignment delta
    var _aligned_len: Int   # length actually mapped (user len + delta)
    var _len: Int           # length the user asked for

    def __init__(out self, base: PTR, ptr: PTR, aligned_len: Int, length: Int):
        self._base = base
        self._ptr = ptr
        self._aligned_len = aligned_len
        self._len = length

    def unmap(deinit self):
        """Release the mapping. Consumes `self` (linear) so it can't be reused."""
        _ = munmap(self._base, self._aligned_len)

    @always_inline
    def ptr(self) -> PTR:
        """Pointer to the caller's requested offset (base + alignment delta).
        Valid until `unmap()` — keep the `Mapping` alive while you use it."""
        return self._ptr

    @always_inline
    def len(self) -> Int:
        return self._len

    def flush(self, flags: SyncFlags = MS_SYNC) raises:
        """Flush the whole mapping to its backing store."""
        if msync(self._base, self._aligned_len, flags) != 0:
            raise Error("msync failed")

    @staticmethod
    def map_file(
        fd: Int32, offset: Int, length: Int, prot: Prot, flags: MapFlags
    ) raises -> Mapping:
        """Map `length` bytes of `fd` starting at an ARBITRARY byte `offset`."""
        var ps = page_size()
        var delta = offset % ps
        var aligned_off = offset - delta
        var aligned_len = length + delta
        var base = mmap(aligned_len, prot, flags, fd, Int64(aligned_off))
        if map_failed(base):
            raise Error("mmap failed")
        var user = PTR(unsafe_from_address=Int(base) + delta)
        return Mapping(base, user, aligned_len, length)

    @staticmethod
    def map_anonymous(length: Int) raises -> Mapping:
        """Anonymous private mapping (no backing file). Offset is 0, so no shim."""
        var base = mmap(
            length, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, Int32(-1))
        if map_failed(base):
            raise Error("anonymous mmap failed")
        return Mapping(base, base, length, length)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: create-or-open a file and map it whole, MAP_SHARED (the IPC path).
# Returns a raw PTR (offset 0 ⇒ no alignment shim needed). The `Mapping` type
# above is the general API; this is the thin whole-region helper the demo uses.
# ─────────────────────────────────────────────────────────────────────────────
def map_shared_file(path: String, size: Int) raises -> PTR:
    var fd = external_call["open", Int32](
        path.unsafe_ptr(), (O_RDWR | O_CREAT).value, Int32(0o666))
    # The open() mode argument is unreliable through external_call (the file can
    # end up 000 and block a second process opening it), so set perms explicitly.
    _ = external_call["fchmod", Int32](fd, Int32(0o666))
    _ = external_call["ftruncate", Int32](fd, Int64(size))
    var addr = mmap(size, PROT_READ | PROT_WRITE, MAP_SHARED, fd)
    _ = external_call["close", Int32](fd)
    if map_failed(addr):
        raise Error("mmap failed for '", path, "'")
    return addr
