#include "graphics/host_gpu/renderer/benchStats.h"

#include "common/debugCounters.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

namespace Libs::Graphics::BenchStats {

namespace {

using Common::DebugCounters::Counter;

std::atomic<uint64_t> g_busy_ns {0};
std::atomic<uint64_t> g_gap_ns {0};

int IntervalSeconds() {
	static const int seconds = [] {
		const char* text = std::getenv("KYTY_BENCH_STATS");
		return text != nullptr ? std::atoi(text) : 0;
	}();
	return seconds;
}

void Report() {
	struct Totals {
		uint64_t frames, busy, gap, wait, faults, submits, draws, dispatches, compile, create;
	};
	const auto read = [] {
		const auto get = [](Counter counter) { return Common::DebugCounters::Get(counter); };
		return Totals {get(Counter::FlipsRequested),   g_busy_ns.load(),
		               g_gap_ns.load(),                get(Counter::GpuWaitNs),
		               get(Counter::GpuFaults),        get(Counter::QueueSubmits),
		               get(Counter::Draws),            get(Counter::Dispatches),
		               get(Counter::ShaderCompileNs),  get(Counter::PipelineCreateNs)};
	};
	const auto seconds = IntervalSeconds();
	auto       last    = read();
	for (;;) {
		std::this_thread::sleep_for(std::chrono::seconds(seconds));
		const auto now    = read();
		const auto frames = now.frames - last.frames;
		const auto per    = frames != 0 ? 1e-6 / static_cast<double>(frames) : 0.0;
		const auto ms     = [per](uint64_t ns) { return static_cast<double>(ns) * per; };
		const auto each   = [frames](uint64_t n) {
            return frames != 0 ? static_cast<double>(n) / static_cast<double>(frames) : 0.0;
		};
		std::printf("bench-stats: frames=%llu fps=%.2f busy=%.1f gap=%.1f wait=%.1f compile=%.1f "
		            "create=%.1f faults=%.0f submits=%.0f draws=%.0f dispatches=%.0f\n",
		            static_cast<unsigned long long>(frames),
		            static_cast<double>(frames) / seconds, ms(now.busy - last.busy),
		            ms(now.gap - last.gap), ms(now.wait - last.wait),
		            ms(now.compile - last.compile), ms(now.create - last.create),
		            each(now.faults - last.faults), each(now.submits - last.submits),
		            each(now.draws - last.draws), each(now.dispatches - last.dispatches));
		std::fflush(stdout);
		last = now;
	}
}

} // namespace

bool Enabled() {
	static const bool enabled = [] {
		if (IntervalSeconds() <= 0) return false;
		Common::DebugCounters::SetEnabled(true);
		std::thread(Report).detach();
		return true;
	}();
	return enabled;
}

void AddGpuTime(uint64_t busy_ns, uint64_t gap_ns) {
	g_busy_ns.fetch_add(busy_ns, std::memory_order_relaxed);
	g_gap_ns.fetch_add(gap_ns, std::memory_order_relaxed);
}

} // namespace Libs::Graphics::BenchStats
