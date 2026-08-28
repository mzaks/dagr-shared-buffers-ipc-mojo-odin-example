# Exercises posix_mmap.mojo — typed flags, page_size(), and the arbitrary-offset
# page-alignment shim. Run: `pixi run mojo run posix_mmap_test.mojo`.
from std.ffi import external_call
from posix_mmap import (
    PTR, Mapping, page_size, map_shared_file, munmap,
    Prot, MapFlags, OFlags,
    PROT_READ, PROT_WRITE, MAP_SHARED, O_RDWR, O_CREAT,
)


def main() raises:
    var ps = page_size()
    print("page_size =", ps)

    # 1) typed flags compose and keep their identity (comptime-folded).
    var rw = PROT_READ | PROT_WRITE
    print("PROT_READ|PROT_WRITE =", rw, "(expect Prot(0x3))")

    # 2) anonymous Mapping — write/read through the handle, then explicit unmap.
    var anon = Mapping.map_anonymous(ps)
    anon.ptr()[] = 123
    print("anon roundtrip =", Int(anon.ptr()[]), "(expect 123)")
    anon^.unmap()

    # 3) THE SHIM: map a file at a NON-page-aligned byte offset, write through the
    #    user pointer, then read the same file offset back via a whole-file map.
    var path = String("/tmp/posix_mmap_shim_test.bin")
    var total = ps * 3
    var off = ps + 64                      # deliberately not page-aligned
    var marker = UInt8(0xAB)

    var fd = external_call["open", Int32](
        path.unsafe_ptr(), (O_RDWR | O_CREAT).value, Int32(0o666))
    _ = external_call["ftruncate", Int32](fd, Int64(total))

    var win = Mapping.map_file(fd, off, 128, PROT_READ | PROT_WRITE, MAP_SHARED)
    # The mapping starts at `off` even though mmap only accepts page-aligned
    # offsets — the shim aligned down to `ps` and returned base + 64.
    print("shim delta ok =", Int(win.ptr()) % ps == 64)
    win.ptr()[] = marker
    win.flush()
    win^.unmap()

    # Verify it landed at file offset `off`, not at the page-aligned base.
    var whole = Mapping.map_file(fd, 0, total, PROT_READ, MAP_SHARED)
    var seen = whole.ptr()[unsafe_offset=off]
    print("byte at offset", off, "=", Int(seen), "(expect", Int(marker), ")")
    var ok = seen == marker
    whole^.unmap()
    _ = external_call["close", Int32](fd)
    if not ok:
        raise Error("SHIM FAILED: byte landed at the wrong file offset")

    # 4) whole-region convenience still returns a raw PTR (the demo path).
    var raw = map_shared_file(String("/tmp/posix_mmap_whole_test.bin"), ps)
    raw[] = 7
    print("whole-region raw ptr roundtrip =", Int(raw[]), "(expect 7)")
    _ = munmap(raw, ps)

    print("ALL OK")
