#include "graphics/host_gpu/renderer/benchStats.h"

#include "common/common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include <windows.h> // IWYU pragma: keep
#include <psapi.h>
#endif

#if defined(KYTY_FUNCTION_PROFILE)
#include <intrin.h>

// Built with -finstrument-functions-after-inlining. KYTY_FN_PROFILE=<start seconds>:<seconds>:<file>
// times every function on the GPU thread. Time in a callee that is not instrumented, such as the
// driver, a GPU wait or JIT code, counts as the caller's own.
#define KYTY_NO_PROFILE __attribute__((no_instrument_function))

namespace {

struct FunctionRow {
	void*    function = nullptr;
	uint64_t calls = 0, self = 0, total = 0;
};
struct FunctionFrame {
	void*    function;
	uint64_t start, children;
};

constexpr size_t RowCount = 1u << 18u;
FunctionRow* g_rows = nullptr;
volatile bool g_profiling = false;
uint64_t      g_stop_tick = 0;
uint64_t      g_start_tick = 0;
const char*   g_profile_path = nullptr;

thread_local bool          t_profiled = false;
thread_local int           t_depth    = 0;
thread_local FunctionFrame t_frames[1024];

KYTY_NO_PROFILE FunctionRow& Row(void* function) {
	auto index = (reinterpret_cast<uint64_t>(function) * 0x9e3779b97f4a7c15ull) >> 46u;
	for (;;) {
		auto& row = g_rows[index];
		if (row.function == function || row.function == nullptr) {
			row.function = function;
			return row;
		}
		index = (index + 1u) & (RowCount - 1u);
	}
}

KYTY_NO_PROFILE void SaveFunctionProfile(uint64_t now) {
	g_profiling = false;
	FILE* file  = std::fopen(g_profile_path, "w");
	if (file == nullptr) return;
	std::fprintf(file, "ticks %llu\n", static_cast<unsigned long long>(now - g_start_tick));
	for (size_t i = 0; i < RowCount; i++) {
		const auto& row = g_rows[i];
		if (row.function != nullptr) {
			std::fprintf(file, "row %llx %llu %llu %llu\n",
			             static_cast<unsigned long long>(reinterpret_cast<uint64_t>(row.function)),
			             static_cast<unsigned long long>(row.calls),
			             static_cast<unsigned long long>(row.self),
			             static_cast<unsigned long long>(row.total));
		}
	}
	std::fclose(file);
}

} // namespace

extern "C" {

KYTY_NO_PROFILE void __cyg_profile_func_enter(void* function, void*) {
	if (!t_profiled || !g_profiling || t_depth >= 1024) return;
	t_frames[t_depth++] = {function, __rdtsc(), 0};
}

KYTY_NO_PROFILE void __cyg_profile_func_exit(void* function, void*) {
	if (!t_profiled || !g_profiling || t_depth == 0) return;
	const auto now = __rdtsc();
	// Frames skipped by an exception or a longjmp are dropped.
	int top = t_depth - 1;
	while (top >= 0 && t_frames[top].function != function) top--;
	if (top < 0) return;
	const auto& frame = t_frames[top];
	const auto  total = now - frame.start;
	auto&       row   = Row(function);
	row.calls++;
	row.self += total - frame.children;
	row.total += total;
	t_depth = top;
	if (top != 0) t_frames[top - 1].children += total;
	if (now >= g_stop_tick) SaveFunctionProfile(now);
}

} // extern "C"
#endif

namespace Libs::Graphics::BenchStats {

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

namespace {

constexpr int      MaxDepth    = 32;
constexpr int      MaxSamples  = 10000;
constexpr LONGLONG PeriodTicks = -10000; // One millisecond, in 100 ns units.

struct Sample {
	int     depth = 0;
	DWORD64 frames[MaxDepth] {};
};

// The target is suspended while this runs, so it must not allocate: that thread may hold the heap
// lock. Frames end at the first code without unwind data, which is most JIT code.
void Walk(HANDLE thread, Sample& sample) {
	CONTEXT context {};
	context.ContextFlags = CONTEXT_FULL;
	if (SuspendThread(thread) == static_cast<DWORD>(-1)) {
		return;
	}
	if (GetThreadContext(thread, &context) != 0) {
		while (sample.depth < MaxDepth && context.Rip != 0) {
			sample.frames[sample.depth++] = context.Rip;
			DWORD64 base                  = 0;
			auto*   function              = RtlLookupFunctionEntry(context.Rip, &base, nullptr);
			if (function == nullptr) {
				break;
			}
			void*   handler = nullptr;
			DWORD64 frame   = 0;
			RtlVirtualUnwind(UNW_FLAG_NHANDLER, base, context.Rip, function, &context, &handler,
			                 &frame, nullptr);
		}
	}
	ResumeThread(thread);
}

void Save(const std::string& path, const Sample* samples, int count) {
	FILE* file = std::fopen(path.c_str(), "w");
	if (file == nullptr) {
		return;
	}
	HMODULE modules[1024];
	DWORD   needed = 0;
	if (EnumProcessModules(GetCurrentProcess(), modules, sizeof(modules), &needed) != 0) {
		for (DWORD i = 0; i < needed / sizeof(HMODULE) && i < 1024; i++) {
			MODULEINFO info {};
			char       name[MAX_PATH] {};
			if (GetModuleInformation(GetCurrentProcess(), modules[i], &info, sizeof(info)) != 0 &&
			    GetModuleFileNameA(modules[i], name, MAX_PATH) != 0) {
				std::fprintf(file, "module %llx %lx %s\n",
				             reinterpret_cast<unsigned long long>(info.lpBaseOfDll),
				             info.SizeOfImage, name);
			}
		}
	}
	for (int i = 0; i < count; i++) {
		std::fputs("sample", file);
		for (int frame = 0; frame < samples[i].depth; frame++) {
			std::fprintf(file, " %llx", static_cast<unsigned long long>(samples[i].frames[frame]));
		}
		std::fputc('\n', file);
	}
	std::fclose(file);
}

void Run(HANDLE thread, int start_seconds, std::string path) {
	Sleep(static_cast<DWORD>(start_seconds) * 1000u);
	auto samples = std::make_unique<Sample[]>(MaxSamples);
	auto timer   = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
	                                      TIMER_ALL_ACCESS);
	for (int i = 0; i < MaxSamples; i++) {
		Walk(thread, samples[i]);
		LARGE_INTEGER due {};
		due.QuadPart = PeriodTicks;
		SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE);
		WaitForSingleObject(timer, INFINITE);
	}
	CloseHandle(timer);
	Save(path, samples.get(), MaxSamples);
	std::printf("gpu-samples: %d saved to %s\n", MaxSamples, path.c_str());
	std::fflush(stdout);
}

} // namespace

void SampleThisThread() {
#if defined(KYTY_FUNCTION_PROFILE)
	if (const char* text = std::getenv("KYTY_FN_PROFILE"); text != nullptr) {
		const int   start   = std::atoi(text);
		const char* colon   = std::strchr(text, ':');
		const int   seconds = colon != nullptr ? std::atoi(colon + 1) : 10;
		g_profile_path      = colon != nullptr ? std::strchr(colon + 1, ':') + 1 : "fn.txt";
		g_rows              = static_cast<FunctionRow*>(std::calloc(RowCount, sizeof(FunctionRow)));
		t_profiled          = true;
		std::thread([start, seconds] {
			Sleep(static_cast<DWORD>(start) * 1000u);
			// Ticks per second, from the clock everything else uses.
			const auto a = __rdtsc();
			Sleep(200);
			const auto rate = (__rdtsc() - a) * 5u;
			g_start_tick    = __rdtsc();
			g_stop_tick     = g_start_tick + rate * static_cast<uint64_t>(seconds);
			g_profiling     = true;
		}).detach();
	}
#endif
	const char* start = std::getenv("KYTY_SAMPLE_GPU");
	const char* path  = std::getenv("KYTY_SAMPLE_OUT");
	if (start == nullptr || path == nullptr) {
		return;
	}
	HANDLE thread = nullptr;
	if (DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &thread, 0,
	                    FALSE, DUPLICATE_SAME_ACCESS) != 0) {
		std::thread(Run, thread, std::atoi(start), std::string(path)).detach();
	}
}

#else

void SampleThisThread() {}

#endif

} // namespace Libs::Graphics::BenchStats
