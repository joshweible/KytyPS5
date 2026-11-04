# Ghost of Yotei: roadmap to 30 fps

The goal is 4K at a smooth 30 fps exactly as on PS5, then 60 fps and beyond. The game keeps seeing a
4K display and its own settings: reducing the work is not an option on this roadmap.

Measured at the campfire scene, 3328x1872, RTX 5090 and Ryzen 7 7800X3D. A frame is 33.3 ms at 30 fps.
The GPU thread and the GPU overlap once the waits between them are gone, so each needs to fit in the
frame on its own. Estimates are from profiles and may not add up exactly: each item is measured when it lands.

| Budget | Now | Target |
|---|---|---|
| GPU work | 55 ms | 28 ms |
| GPU thread recording | 75 ms | 28 ms |
| GPU thread blocked on the GPU | 33 ms | under 5 ms |

## Order of work

1. **Precise compute barriers.** Every dispatch pays full barriers, about 2,190 a frame. Removing
   them outright (unsafe) took GPU time to 36 ms, so keep only the ones real conflicts need.
2. **Recording costs with one owner.** Walker faults (8 ms), texture lookups (9 ms), VRAM copies on
   the GPU thread (5 ms), memory protection calls (3 ms), buffer sync and BDA (11 ms).
3. **Draw barriers and render pass breaks.** The same analysis for graphics.
4. **Shader quality.** Wave64 doubles shader code, not runtime, so doubling itself is not the cost.
   Look at bounds checks, flattened control flow and the heaviest passes shader by shader.
5. **Parallel preparation.** Move shader and descriptor preparation to workers ahead of recording.
6. **Game thread faults.** About 3,200 write faults a frame, plus false sharing on reads.
7. **Pipeline delivery.** Remaining variants (buffer strides, candidate counts) and precompiling.
8. **Check the real game.** Open world and combat, the far terrain bug, and 30 fps held there.

Milestones: 15 fps after 1 and 2, 20 fps after 3 and 4, 30 fps after 5 to 7.
