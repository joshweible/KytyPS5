#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SPIRVLOOPGUARD_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SPIRVLOOPGUARD_H_

#include <cstdint>
#include <vector>

namespace Libs::Graphics {

// Debug tool for a shader that hangs the GPU. Each guarded loop of the entry function counts its
// iterations per invocation and returns past `limit`, so the device survives; which loops are
// guarded lets a search find the one that never ends. The result differs from the guest shader.
std::vector<uint32_t> GuardLoops(const std::vector<uint32_t>& spirv, uint32_t first, uint32_t last,
                                 uint32_t limit);

// KYTY_LOOP_GUARD=<hash>[:first:last[:limit]]: guards that shader's module; others pass unchanged.
std::vector<uint32_t> ApplyLoopGuard(uint64_t shader_hash, const std::vector<uint32_t>& spirv);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SPIRVLOOPGUARD_H_
