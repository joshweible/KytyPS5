#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BENCHSTATS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BENCHSTATS_H_

#include <cstdint>

namespace Libs::Graphics::BenchStats {

// KYTY_BENCH_STATS=<seconds>: prints frame rate and where each frame's time went on that interval,
// for tools/yotei_bench.ps1. GPU busy time is the union of command buffer intervals; a gap is GPU
// time with no command buffer executing. Off by default, and costs nothing while off.
[[nodiscard]] bool Enabled();
void               AddGpuTime(uint64_t busy_ns, uint64_t gap_ns);
// One guest fault on GPU-tracked memory, timed by where it spent its time. Runs in an exception
// handler, so it only touches a fixed table.
// One read of memory the GPU may have written: time queued for the GPU thread, time waiting for
// the GPU, time publishing the data, bytes copied (0: nothing was GPU-written).
void SetFaultPc(uint64_t pc); // The faulting instruction, for the read this thread reports next.
[[nodiscard]] uint64_t FaultPc();
void NoteRead(uint64_t address, bool write, bool gpu_thread, uint64_t queue_ns, uint64_t wait_ns,
              uint64_t copy_ns, uint64_t bytes);
void NoteFault(uint64_t address, bool write, bool gpu_thread, uint64_t buffer_ns,
               uint64_t texture_ns, uint64_t heap_ns);

} // namespace Libs::Graphics::BenchStats

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BENCHSTATS_H_
