#include "graphics/host_gpu/renderer/masterSemaphore.h"

#include "common/assert.h"
#include "common/debugCounters.h"
#include "graphics/host_gpu/graphicContext.h"

#include <algorithm>
#include <array>
#include <map>
#include <utility>
#include <atomic>
#include <cstdint>
#include <cstdio>
namespace Libs::Graphics {

// Newest last: which dispatches went to the GPU before a device loss.
static std::array<std::array<uint64_t, 4>, 65536> g_recent_dispatches {};
static std::atomic<uint64_t>                   g_recent_dispatch_count {0};

void NoteDispatch(uint64_t hash, uint32_t x, uint32_t y, uint32_t z, uint64_t tick) {
	const auto index = g_recent_dispatch_count.fetch_add(1u);
	g_recent_dispatches[index % g_recent_dispatches.size()] = {hash, uint64_t {x} << 42u | uint64_t {y} << 21u | z, tick, 0u};
}

static void PrintRecentDispatches(uint64_t waited_tick) {
	const auto count = g_recent_dispatch_count.load();
	std::printf("device lost waiting for tick %llu\n", static_cast<unsigned long long>(waited_tick));
	std::map<std::pair<uint64_t, uint64_t>, uint32_t> seen;
	std::map<std::pair<uint64_t, uint64_t>, uint64_t> largest;
	for (uint64_t i = count > g_recent_dispatches.size() ? count - g_recent_dispatches.size() : 0u;
	     i < count; i++) {
		const auto& d = g_recent_dispatches[i % g_recent_dispatches.size()];
		if (d[2] + 3u < waited_tick) continue;
		seen[{d[2], d[0]}]++;
		const auto groups = (d[1] >> 42u) * ((d[1] >> 21u) & 0x1fffffu) * (d[1] & 0x1fffffu);
		auto&      most   = largest[{d[2], d[0]}];
		most              = std::max(most, groups);
	}
	for (const auto& [key, times]: seen) {
		std::printf("  tick=%llu cs=%016llx x%u max_groups=%llu\n",
		            static_cast<unsigned long long>(key.first),
		            static_cast<unsigned long long>(key.second), times,
		            static_cast<unsigned long long>(largest[key]));
	}
	std::fflush(stdout);
}

MasterSemaphore::MasterSemaphore(GraphicContext& graphics): m_graphics(graphics) {
	vk::SemaphoreTypeCreateInfo type_info {};
	type_info.semaphoreType = vk::SemaphoreType::eTimeline;
	type_info.initialValue  = 0;

	vk::SemaphoreCreateInfo create_info {};
	create_info.pNext = &type_info;

	const auto result = m_graphics.device.createSemaphore(&create_info, nullptr, &m_semaphore);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_semaphore == nullptr);
}

MasterSemaphore::~MasterSemaphore() {
	m_graphics.device.destroySemaphore(m_semaphore, nullptr);
}

void MasterSemaphore::Refresh() {
	uint64_t   counter = 0;
	const auto result  = m_graphics.device.getSemaphoreCounterValue(m_semaphore, &counter);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	auto known = m_gpu_tick.load(std::memory_order_acquire);
	while (known < counter &&
	       !m_gpu_tick.compare_exchange_weak(known, counter, std::memory_order_release,
	                                         std::memory_order_relaxed)) {
	}
}

void MasterSemaphore::Wait(uint64_t tick) {
	if (IsFree(tick)) {
		return;
	}
	Refresh();
	if (IsFree(tick)) {
		return;
	}

	vk::SemaphoreWaitInfo wait_info {};
	wait_info.semaphoreCount = 1;
	wait_info.pSemaphores    = &m_semaphore;
	wait_info.pValues        = &tick;

	Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::GpuWaits,
	                                         Common::DebugCounters::Counter::GpuWaitNs);
	const auto result = m_graphics.device.waitSemaphores(&wait_info, UINT64_MAX);
	if (result != vk::Result::eSuccess) {
		std::printf("GPU wait failed: %s\n", vk::to_string(result).c_str());
		PrintRecentDispatches(tick);
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
	Refresh();
}

} // namespace Libs::Graphics
