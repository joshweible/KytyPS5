#include "graphics/host_gpu/renderer/benchStats.h"

#include "common/debugCounters.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

namespace Libs::Graphics::BenchStats {

namespace {

using Common::DebugCounters::Counter;

std::atomic<uint64_t> g_busy_ns {0};
std::atomic<uint64_t> g_gap_ns {0};

// Faults by 1 MiB region, access and thread. Fixed size: a full table counts in `overflow`.
struct FaultRow {
	std::atomic<uint64_t> key {0}; // Region << 2 | write << 1 | gpu_thread, plus one so 0 is free.
	std::atomic<uint64_t> count {0};
	std::atomic<uint64_t> ns {0};
};
constexpr size_t      FaultRows = 4096;
FaultRow              g_faults[FaultRows];
std::atomic<uint64_t> g_fault_overflow {0};
std::atomic<uint64_t> g_fault_buffer_ns {0};
std::atomic<uint64_t> g_fault_texture_ns {0};
std::atomic<uint64_t> g_fault_heap_ns {0};
std::atomic<uint64_t> g_fault_kinds[4] {};    // Count by write << 1 | gpu_thread.
std::atomic<uint64_t> g_fault_kind_ns[4] {};

struct ReadRow {
	std::atomic<uint64_t> key {0};
	std::atomic<uint64_t> count {0};
	std::atomic<uint64_t> empty {0}; // Found nothing GPU-written.
	std::atomic<uint64_t> queue_ns {0};
	std::atomic<uint64_t> wait_ns {0};
	std::atomic<uint64_t> copy_ns {0};
	std::atomic<uint64_t> bytes {0};
	std::atomic<uint64_t> pc {0}; // Last faulting instruction.
};
ReadRow g_reads[FaultRows];
std::atomic<uint64_t> g_writer_finished {0};
std::atomic<uint64_t> g_writer_pending {0};
thread_local uint64_t t_fault_pc = 0;

std::atomic<uint64_t> g_thread_ns[GpuThreadTimes];
std::atomic<uint64_t> g_write_kinds[WriteKinds];
std::atomic<uint64_t> g_clean_reads {0};
std::atomic<uint64_t> g_image_known {0};
std::atomic<uint64_t> g_image_searched {0};
std::atomic<uint64_t> g_dispatch_runs {0};
std::atomic<uint64_t> g_dispatch_conflicts {0};
std::atomic<uint64_t> g_dispatch_overlaps {0};
std::atomic<uint64_t> g_dispatch_why[7];

void ReportReads(double frames) {
	struct Row {
		uint64_t key, count, empty, queue, wait, copy, bytes, pc;
	};
	std::vector<Row> rows;
	Row              total {};
	for (auto& row: g_reads) {
		const Row r {row.key.load(std::memory_order_relaxed),
		            row.count.exchange(0, std::memory_order_relaxed),
		            row.empty.exchange(0, std::memory_order_relaxed),
		            row.queue_ns.exchange(0, std::memory_order_relaxed),
		            row.wait_ns.exchange(0, std::memory_order_relaxed),
		            row.copy_ns.exchange(0, std::memory_order_relaxed),
		            row.bytes.exchange(0, std::memory_order_relaxed),
		            row.pc.load(std::memory_order_relaxed)};
		if (r.key == 0 || r.count == 0) continue;
		total.count += r.count;
		total.empty += r.empty;
		total.queue += r.queue;
		total.wait += r.wait;
		total.copy += r.copy;
		total.bytes += r.bytes;
		rows.push_back(r);
	}
	std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) {
		return a.queue + a.wait + a.copy > b.queue + b.wait + b.copy;
	});
	const auto ms = [frames](uint64_t ns) { return static_cast<double>(ns) * 1e-6 / frames; };
	const auto per = [frames](uint64_t n) { return static_cast<double>(n) / frames; };
	std::printf("read-total: %.0f/frame empty=%.0f queue=%.1f wait=%.1f copy=%.1f ms/frame "
	            "%.2f MiB/frame\n",
	            per(total.count), per(total.empty), ms(total.queue), ms(total.wait),
	            ms(total.copy), per(total.bytes) / (1024.0 * 1024.0));
	std::printf("gpu-thread: idle=%.1f draw=%.1f dispatch=%.1f other=%.1f ms/frame\n",
	            static_cast<double>(g_thread_ns[GpuIdle].exchange(0)) * 1e-6 / frames,
	            static_cast<double>(g_thread_ns[GpuDraw].exchange(0)) * 1e-6 / frames,
	            static_cast<double>(g_thread_ns[GpuDispatch].exchange(0)) * 1e-6 / frames,
	            static_cast<double>(g_thread_ns[GpuOther].exchange(0)) * 1e-6 / frames);
	std::printf("read-kind: binding=%.1f fill=%.1f copy=%.1f image=%.1f /frame\n",
	            per(g_write_kinds[WriteBinding].exchange(0)), per(g_write_kinds[WriteFill].exchange(0)),
	            per(g_write_kinds[WriteCopy].exchange(0)), per(g_write_kinds[WriteImage].exchange(0)));
	std::printf("image-lookup: known=%.0f searched=%.0f /frame\n", per(g_image_known.exchange(0)),
	            per(g_image_searched.exchange(0)));
	std::printf("read-clean: %.1f /frame\n", per(g_clean_reads.exchange(0)));
	std::printf("dispatch-why: seen-raw-write=%.0f raw-write=%.0f raw-read=%.0f seen-raw-read=%.0f heap=%.0f full=%.0f overlap=%.0f /frame\n",
	            per(g_dispatch_why[0].exchange(0)), per(g_dispatch_why[1].exchange(0)), per(g_dispatch_why[2].exchange(0)),
	            per(g_dispatch_why[3].exchange(0)), per(g_dispatch_why[4].exchange(0)), per(g_dispatch_why[5].exchange(0)),
	            per(g_dispatch_why[6].exchange(0)));
	std::printf("dispatch-order: runs=%.1f conflicts=%.1f overlaps=%.1f /frame\n",
	            per(g_dispatch_runs.exchange(0)), per(g_dispatch_conflicts.exchange(0)),
	            per(g_dispatch_overlaps.exchange(0)));
	std::printf("read-writer: finished=%.1f pending=%.1f /frame\n",
	            per(g_writer_finished.exchange(0)), per(g_writer_pending.exchange(0)));
	std::printf("bench-anchor: 0x%llx\n", reinterpret_cast<unsigned long long>(&ReportReads));
	const char* const names[4] = {"read-game", "read-gpu", "write-game", "write-gpu"};
	for (size_t i = 0; i < std::min<size_t>(rows.size(), 14u); i++) {
		const auto& r   = rows[i];
		const auto  key = r.key - 1u;
		std::printf("read-top: 0x%012llx %-10s %.1f/frame empty=%.1f queue=%.2f wait=%.2f copy=%.2f "
		            "ms/frame %.0f KiB/frame pc=0x%llx\n",
		            static_cast<unsigned long long>((key >> 2u) << 20u), names[key & 3u],
		            per(r.count), per(r.empty), ms(r.queue), ms(r.wait), ms(r.copy),
		            per(r.bytes) / 1024.0, static_cast<unsigned long long>(r.pc));
	}
}

void ReportFaults(double frames) {
	struct Row {
		uint64_t key, count, ns;
	};
	std::vector<Row> rows;
	for (auto& row: g_faults) {
		const auto key   = row.key.load(std::memory_order_relaxed);
		const auto count = row.count.exchange(0, std::memory_order_relaxed);
		const auto ns    = row.ns.exchange(0, std::memory_order_relaxed);
		if (key != 0 && count != 0) rows.push_back({key - 1u, count, ns});
	}
	std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.ns > b.ns; });
	const char* const names[4] = {"read-game", "read-gpu", "write-game", "write-gpu"};
	std::printf("fault-stats: buffer=%.1f texture=%.1f heap=%.1f ms/frame overflow=%llu\n",
	            static_cast<double>(g_fault_buffer_ns.exchange(0)) * 1e-6 / frames,
	            static_cast<double>(g_fault_texture_ns.exchange(0)) * 1e-6 / frames,
	            static_cast<double>(g_fault_heap_ns.exchange(0)) * 1e-6 / frames,
	            static_cast<unsigned long long>(g_fault_overflow.exchange(0)));
	for (int kind = 0; kind < 4; kind++) {
		std::printf("fault-kind: %-10s %.0f/frame %.1f ms/frame\n", names[kind],
		            static_cast<double>(g_fault_kinds[kind].exchange(0)) / frames,
		            static_cast<double>(g_fault_kind_ns[kind].exchange(0)) * 1e-6 / frames);
	}
	for (size_t i = 0; i < std::min<size_t>(rows.size(), 12u); i++) {
		const auto& row = rows[i];
		std::printf("fault-top: 0x%012llx %-10s %.0f/frame %.2f ms/frame\n",
		            static_cast<unsigned long long>((row.key >> 2u) << 20u), names[row.key & 3u],
		            static_cast<double>(row.count) / frames,
		            static_cast<double>(row.ns) * 1e-6 / frames);
	}
}

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
		if (frames != 0) {
			ReportFaults(static_cast<double>(frames));
			ReportReads(static_cast<double>(frames));
		}
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

void NoteWriter(bool finished) {
	(finished ? g_writer_finished : g_writer_pending).fetch_add(1u, std::memory_order_relaxed);
}

void SetFaultPc(uint64_t pc) {
	t_fault_pc = pc;
}

uint64_t FaultPc() {
	return t_fault_pc;
}

void NoteRead(uint64_t address, bool write, bool gpu_thread, uint64_t queue_ns, uint64_t wait_ns,
              uint64_t copy_ns, uint64_t bytes) {
	const auto pc   = t_fault_pc;
	const auto kind = (write ? 2u : 0u) | (gpu_thread ? 1u : 0u);
	const auto key  = ((address >> 20u) << 2u | kind) + 1u;
	for (size_t probe = 0; probe < 64u; probe++) {
		auto&    row      = g_reads[(key * 0x9e3779b97f4a7c15ull >> 52u) + probe & (FaultRows - 1u)];
		uint64_t expected = 0;
		if (row.key.compare_exchange_strong(expected, key, std::memory_order_relaxed) ||
		    expected == key) {
			row.count.fetch_add(1u, std::memory_order_relaxed);
			row.empty.fetch_add(bytes == 0u ? 1u : 0u, std::memory_order_relaxed);
			row.queue_ns.fetch_add(queue_ns, std::memory_order_relaxed);
			row.wait_ns.fetch_add(wait_ns, std::memory_order_relaxed);
			row.copy_ns.fetch_add(copy_ns, std::memory_order_relaxed);
			row.bytes.fetch_add(bytes, std::memory_order_relaxed);
			row.pc.store(pc, std::memory_order_relaxed);
			return;
		}
	}
}

void NoteFault(uint64_t address, bool write, bool gpu_thread, uint64_t buffer_ns,
               uint64_t texture_ns, uint64_t heap_ns) {
	const auto total = buffer_ns + texture_ns + heap_ns;
	const auto kind  = (write ? 2u : 0u) | (gpu_thread ? 1u : 0u);
	const auto key   = ((address >> 20u) << 2u | kind) + 1u;
	g_fault_buffer_ns.fetch_add(buffer_ns, std::memory_order_relaxed);
	g_fault_texture_ns.fetch_add(texture_ns, std::memory_order_relaxed);
	g_fault_heap_ns.fetch_add(heap_ns, std::memory_order_relaxed);
	g_fault_kinds[kind].fetch_add(1u, std::memory_order_relaxed);
	g_fault_kind_ns[kind].fetch_add(total, std::memory_order_relaxed);
	for (size_t probe = 0; probe < 64u; probe++) {
		auto&    row      = g_faults[(key * 0x9e3779b97f4a7c15ull >> 52u) + probe & (FaultRows - 1u)];
		uint64_t expected = 0;
		if (row.key.compare_exchange_strong(expected, key, std::memory_order_relaxed) ||
		    expected == key) {
			row.count.fetch_add(1u, std::memory_order_relaxed);
			row.ns.fetch_add(total, std::memory_order_relaxed);
			return;
		}
	}
	g_fault_overflow.fetch_add(1u, std::memory_order_relaxed);
}

void AddGpuThreadTime(GpuThreadTime kind, uint64_t ns) {
	g_thread_ns[kind].fetch_add(ns, std::memory_order_relaxed);
}

void NoteDispatchBarrier(bool conflict) {
	(conflict ? g_dispatch_conflicts : g_dispatch_runs).fetch_add(1u, std::memory_order_relaxed);
}

void NoteDispatchWhy(uint32_t why, uint64_t) {
	g_dispatch_why[why].fetch_add(1u, std::memory_order_relaxed);
}

void NoteDispatchOverlap() {
	g_dispatch_overlaps.fetch_add(1u, std::memory_order_relaxed);
}

void NoteImageLookup(bool known) {
	(known ? g_image_known : g_image_searched).fetch_add(1u, std::memory_order_relaxed);
}

void NoteCleanRead() {
	g_clean_reads.fetch_add(1u, std::memory_order_relaxed);
}

void NoteWriteKind(WriteKind kind) {
	g_write_kinds[kind].fetch_add(1u, std::memory_order_relaxed);
}

void AddGpuTime(uint64_t busy_ns, uint64_t gap_ns) {
	g_busy_ns.fetch_add(busy_ns, std::memory_order_relaxed);
	g_gap_ns.fetch_add(gap_ns, std::memory_order_relaxed);
}

} // namespace Libs::Graphics::BenchStats
