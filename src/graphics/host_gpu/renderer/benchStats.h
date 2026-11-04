#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BENCHSTATS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BENCHSTATS_H_

#include <cstdint>

namespace Libs::Graphics::BenchStats {

// KYTY_BENCH_STATS=<seconds>: prints frame rate and where each frame's time went on that interval,
// for tools/yotei_bench.ps1. GPU busy time is the union of command buffer intervals; a gap is GPU
// time with no command buffer executing. Off by default, and costs nothing while off.
[[nodiscard]] bool Enabled();
void               AddGpuTime(uint64_t busy_ns, uint64_t gap_ns);

} // namespace Libs::Graphics::BenchStats

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BENCHSTATS_H_
