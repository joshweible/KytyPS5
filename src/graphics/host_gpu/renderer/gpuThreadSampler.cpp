#include "graphics/host_gpu/renderer/benchStats.h"

#include "common/common.h"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include <windows.h> // IWYU pragma: keep
#include <psapi.h>
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
