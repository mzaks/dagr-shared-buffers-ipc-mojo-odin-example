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
import "vendor:raylib/rlgl"
import fsb "forest_sb"

// Window size, forest depth, and the shared-region path all come from the schema
// (imported from the generated `forest_sb` overlay) — one source of truth shared
// with the Mojo producer, nothing duplicated here.
WIN_W :: i32(fsb.WORLD_W)
WIN_H :: i32(fsb.WORLD_H)

// Wait until the producer has created and sized the shared file, then mmap it RW
// (the double_buffer consumer mutates the control word, so it needs write access).
attach :: proc() -> [^]u8 {
	waited := 0
	for {
		if info, err := os.stat(fsb.PATH, context.temp_allocator); err == nil && info.size >= i64(fsb.BYTE_SIZE) {
			break
		}
		free_all(context.temp_allocator)
		time.sleep(20 * time.Millisecond)
		waited += 20
		if waited > 15_000 {
			fmt.eprintln("renderer: timed out waiting for the producer to create", fsb.PATH)
			os.exit(1)
		}
	}
	fd := posix.open(fsb.PATH, {.RDWR})
	if fd < 0 {
		fmt.eprintln("renderer: cannot open", fsb.PATH)
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

HAZE :: rl.Color{40, 52, 70, 255} // misty blue-grey the distance fades toward
// The producer packs BRANCHES segments per tree, contiguously, with tree 0 the
// farthest (so the array runs back-to-front). The renderer re-derives which tree a
// segment belongs to from its index — TREES and TREE_DEPTH are shared schema
// constants — and fades the whole tree toward the haze colour by its distance.
BRANCHES :: (1 << uint(fsb.TREE_DEPTH)) - 1

// Colour a branch by depth (trunk brown → canopy green), then apply atmospheric
// perspective: the farther the tree, the more it fades into the haze.
branch_color :: proc(depth: int, seg: int) -> rl.Color {
	t := f32(depth) / f32(fsb.TREE_DEPTH)
	r := f32(110) * (1 - t) + 70 * t
	g := f32(70) * (1 - t) + 200 * t
	b := f32(45) * (1 - t) + 90 * t
	dist := 1 - f32(seg / BRANCHES) / f32(fsb.TREES - 1) // 1 = far, 0 = near
	h := dist * 0.82
	return rl.Color{
		u8(r * (1 - h) + f32(HAZE.r) * h),
		u8(g * (1 - h) + f32(HAZE.g) * h),
		u8(b * (1 - h) + f32(HAZE.b) * h),
		255,
	}
}

// Depths below this are the trunk + major limbs (a few thousand segments): drawn as
// thick anti-aliased quads. Everything deeper — the fine mass, ~95% of the forest —
// is streamed as 1px GL lines through rlgl, which is far cheaper per segment and
// scales to hundreds of thousands of branches at well over 60 fps. rlgl's internal
// batch buffer is finite, so the line stream is flushed every GL_LINE_CHUNK lines
// (End → DrawRenderBatchActive → Begin) — otherwise a large forest silently drops
// every branch past the first bufferful.
THICK_MAX_DEPTH :: 8
GL_LINE_CHUNK :: 8000

draw_forest :: proc(f: fsb.Forest, n: int) {
	// pass 1 — thick trunks + major limbs, drawn first so the canopy sits over them.
	for i in 0 ..< n {
		depth := int(fsb.forest_depth(f, i))
		if depth >= THICK_MAX_DEPTH {
			continue
		}
		y0 := fsb.forest_y0(f, i)
		start := rl.Vector2{fsb.forest_x0(f, i), y0}
		end := rl.Vector2{fsb.forest_x1(f, i), fsb.forest_y1(f, i)}
		rl.DrawLineEx(start, end, f32(fsb.TREE_DEPTH - depth) * 0.7, branch_color(depth, i))
	}
	// pass 2 — the fine branches, batched as one GL_LINES stream (chunk-flushed).
	rlgl.Begin(rlgl.LINES)
	pending := 0
	for i in 0 ..< n {
		depth := int(fsb.forest_depth(f, i))
		if depth < THICK_MAX_DEPTH {
			continue
		}
		y0 := fsb.forest_y0(f, i)
		col := branch_color(depth, i)
		rlgl.Color4ub(col.r, col.g, col.b, 255)
		rlgl.Vertex2f(fsb.forest_x0(f, i), y0)
		rlgl.Vertex2f(fsb.forest_x1(f, i), fsb.forest_y1(f, i))
		pending += 1
		if pending >= GL_LINE_CHUNK {
			rlgl.End()
			rlgl.DrawRenderBatchActive() // flush so the buffer can't overflow
			rlgl.Begin(rlgl.LINES)
			pending = 0
		}
	}
	rlgl.End()
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

		draw_forest(f, n)

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
