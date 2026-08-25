// Fractal-forest renderer (process B of the IPC demo).
//
// Attaches to the SAME mmap'd region the Mojo GPU producer publishes into, and
// draws the newest completed frame with raylib at 60 fps. It reads the geometry
// straight out of the shared bytes through the generated Dagr overlay (`forest_sb`)
// — pure offset arithmetic, no deserialization — and never blocks the producer:
// `consumer_read` latches the newest published `double_buffer` slot with one atomic
// exchange and never tears or retries.
//
//   Mojo (GPU) --commit(atomic)--> mmap region --consumer_read()--> Odin + raylib
package main

import "core:fmt"
import "core:os"
import "core:sys/posix"
import "core:time"
import rl "vendor:raylib"
import fsb "forest_sb"

WIN_W :: 1280
WIN_H :: 800
TREE_DEPTH :: 12 // must match forest_schema.py / producer.mojo
PATH :: "/tmp/dagr_forest_ipc.bin"

// Wait until the producer has created and sized the shared file, then mmap it RW
// (the double_buffer consumer mutates the control word, so it needs write access).
attach :: proc() -> [^]u8 {
	waited := 0
	for {
		if info, err := os.stat(PATH, context.temp_allocator); err == nil && info.size >= i64(fsb.BYTE_SIZE) {
			break
		}
		free_all(context.temp_allocator)
		time.sleep(20 * time.Millisecond)
		waited += 20
		if waited > 15_000 {
			fmt.eprintln("renderer: timed out waiting for the producer to create", PATH)
			os.exit(1)
		}
	}
	fd := posix.open(PATH, {.RDWR})
	if fd < 0 {
		fmt.eprintln("renderer: cannot open", PATH)
		os.exit(1)
	}
	addr := posix.mmap(nil, uint(fsb.BYTE_SIZE), {.READ, .WRITE}, {.SHARED}, fd, 0)
	posix.close(fd)
	if addr == posix.MAP_FAILED {
		fmt.eprintln("renderer: mmap failed")
		os.exit(1)
	}
	return cast([^]u8)addr
}

// Colour a branch by depth: trunk brown → canopy green, with a little lift so the
// tips read as foliage.
branch_color :: proc(depth: int) -> rl.Color {
	t := f32(depth) / f32(TREE_DEPTH)
	r := u8(110 * (1 - t) + 70 * t)
	g := u8(70 * (1 - t) + 200 * t)
	b := u8(45 * (1 - t) + 90 * t)
	return rl.Color{r, g, b, 255}
}

main :: proc() {
	region := attach()
	consumer := fsb.consumer_from_raw(region)

	rl.SetConfigFlags({.MSAA_4X_HINT})
	rl.InitWindow(WIN_W, WIN_H, "Dagr SharedBuffer IPC — Mojo GPU → Odin raylib")
	rl.SetTargetFPS(60)

	for !rl.WindowShouldClose() {
		f := fsb.consumer_read(&consumer) // latch newest published frame (atomic)
		n := int(fsb.forest_count(f))
		frame := fsb.forest_frame(f)

		rl.BeginDrawing()
		rl.ClearBackground(rl.Color{14, 16, 22, 255})

		for i in 0 ..< n {
			depth := int(fsb.forest_depth(f, i))
			start := rl.Vector2{fsb.forest_x0(f, i), fsb.forest_y0(f, i)}
			end := rl.Vector2{fsb.forest_x1(f, i), fsb.forest_y1(f, i)}
			thick := max(f32(1.0), f32(TREE_DEPTH - depth) * 0.7)
			rl.DrawLineEx(start, end, thick, branch_color(depth))
		}

		rl.DrawText(
			rl.TextFormat("Mojo GPU  ->  Dagr SharedBuffer (mmap, double_buffer)  ->  Odin raylib"),
			16, 14, 20, rl.RAYWHITE,
		)
		rl.DrawText(
			rl.TextFormat("segments: %d   producer frame: %d   fps: %d", n, frame, rl.GetFPS()),
			16, 40, 18, rl.Color{150, 170, 190, 255},
		)
		rl.EndDrawing()
	}

	rl.CloseWindow()
}
