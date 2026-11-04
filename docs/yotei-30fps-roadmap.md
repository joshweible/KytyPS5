# Ghost of Yotei: roadmap to 30 fps

The goal is 4K at a smooth 30 fps exactly as on PS5, then 60 fps and beyond. The game keeps seeing a
4K display and its own settings: reducing the work is not an option on this roadmap.

Measured at the campfire scene, 3328x1872, RTX 5090 and Ryzen 7 7800X3D. A frame is 33.3 ms at 30 fps.
The GPU thread and the GPU overlap once the waits between them are gone, so each needs to fit in the
frame on its own. Each item is measured when it lands.

| Budget | Start | Now | Target |
|---|---|---|---|
| Frame | 122 ms | 79 ms | 33 ms |
| GPU work | 55 ms | 46 ms | 28 ms |
| GPU thread recording | 75 ms | 68 ms | 28 ms |
| GPU thread blocked on the GPU | 33 ms | 9 ms | under 5 ms |

## Done

- Direct reads from CPU-visible VRAM, with waits on the game's thread.
- Reads of clean bytes on pages the GPU also wrote, without a wait or a page fault.
- GPU-written SRT constants reach shaders by a GPU copy, not a CPU wait.
- Compute barriers only between dispatches that conflict.
- One shader per table offset, and graphics pipelines built off the GPU thread.
- Compiler work is measured and worth about 5%: ThinLTO with Zen 4 tuning gives 10.88 fps against
  10.48, and PGO 11.0. AVX-512 is worse, because 512-bit streaming loads from VRAM are very slow on
  Zen 4. ThinLTO and PGO together fail to link. None of this is in the default build yet.

## Measured and ruled out

The GPU thread limits the frame, so GPU savings do not raise the frame rate until recording drops
below GPU time. Results at the campfire, with run-to-run noise of about 0.3 fps:

- Barriers cost about 1.3 ms of direct GPU time. Following the game's own sync packets cut conflict
  barriers from 1,110 to 180 a frame and made GPU time worse.
- Clocks are at full speed in P0, at 2850 MHz, with PCIe 4.0 x16.
- Robust image access makes no difference, and relaxing float contraction gains 3% at most.
- None of the 40 heaviest shaders use the dispatcher fallback.
- Exec selects are spread over many masks, so removing the root mask would save under 10%.
- Writing uploads straight into mapped VRAM loses the device: the CPU gets ahead of queued GPU work.
- Removing the upload barrier changes nothing.
- A thread per guest ring saves 15 ms at most: ring 0 holds 64 ms of recording.
- COND_EXEC's 6 ms is a real cross-ring sync, guarding waits for async compute.

## Tools

- `tools/yotei_bench.ps1`: frame rate, stage, packet and event counters.
- `KYTY_GPU_PROFILE`: GPU time by shader, barrier and other work.
- `KYTY_FUNCTION_PROFILE` builds with `tools/function_report.py`: exact time per function on the GPU
  thread. The hooks add about 18 ms, mostly to functions called thousands of times a frame, so
  rank those by call count before believing them.

## Order of work

1. **Recording: the structural problem.** It is flat by function and by shader, about 15 us per
   operation, so caches alone will not reach 28 ms. The natural cut between preparing and recording
   an operation is blocked by recording reading live registers: copying them per draw costs too
   much, so this needs a register delta stream or a different split. This is the largest remaining item.
2. **Recording: costs with one owner.** Binding reuse across lookups (about 32,000 a frame), shader
   lookup, buffer sync, memory protection calls.
3. **Draw barriers and render pass breaks.** The compute analysis, applied to graphics.
4. **Shader quality.** Wave64 doubles shader code, not runtime. Look at bounds checks, flattened
   control flow and the heaviest passes shader by shader.
5. **Remaining compute conflicts.** About 1,100 a frame from overlapping writable bindings.
6. **Game thread faults.** About 4,000 write faults a frame, plus false sharing on game reads.
7. **Remaining GPU waits.** COND_EXEC reads a GPU value on the CPU, 6.6 ms a frame.
8. **Pipeline delivery.** Variants by buffer stride and candidate count, and precompiling.
9. **Check the real game.** Open world and combat, the far terrain bug, and 30 fps held there.
