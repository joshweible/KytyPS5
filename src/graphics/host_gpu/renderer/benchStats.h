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
enum GpuThreadTime : uint32_t {
	GpuIdle,
	GpuDraw,
	GpuDispatch,
	GpuOther,
	GpuFrontActive, // Reading packets, including GpuFrontWait.
	GpuFrontWait,   // Waiting for the GPU thread.
	GpuCommands,    // Work other threads queued for the GPU thread, such as guest memory reads.
	GpuThreadTimes
};
void AddGpuThreadTime(GpuThreadTime kind, uint64_t ns);
enum WriteKind : uint32_t { WriteBinding, WriteFill, WriteCopy, WriteImage, WriteKinds };
void NoteWriteKind(WriteKind kind); // The kind of write behind a read that had to wait.
// KYTY_SAMPLE_GPU=<start seconds> KYTY_SAMPLE_OUT=<path>: samples this thread's stack at 1 kHz
// for ten seconds, for tools/sample_report.py.
void SampleThisThread();
void NoteCleanRead(); // A walker load served without the GPU from a page it wrote part of.
void NoteDispatchBarrier(bool conflict); // Run started, or a conflict inside one.
void NoteDispatchOverlap();
void NoteDispatchWhy(uint32_t why, uint64_t hash);
void NoteImageLookup(bool known);
void NotePacket(uint32_t opcode, uint64_t ns);
void NoteSyncPacket(uint32_t opcode); // The front thread waited for the GPU thread to run it.
void NoteShader(uint64_t hash); // The operation the current packet recorded.
void AddSchedulerWait(uint64_t ns);
void NoteLateWords(uint32_t count); // Flat words filled in by a GPU copy.
// Charges the time since the last mark on this thread to a stage of recording.
enum RecordStage : uint32_t {
	StageNone,
	DrawChecks,
	DrawPrograms,
	DrawTargets,
	DrawBindings,
	DrawImages,
	DrawBuffers,
	DrawPipeline,
	DrawAcquire,
	DrawCommit,
	DrawRecord,
	DispatchPrograms,
	DispatchBindings,
	DispatchBda,
	DispatchImages,
	DispatchBuffers,
	DispatchCommit,
	DispatchRecord,
	RecordStages
};
void MarkStage(RecordStage stage);
void SetStageBank(uint32_t bank);
enum GpuEvent : uint32_t {
	EventRenderPass,
	EventImageBarrier,
	EventImageUpload,
	EventBufferUpload,
	EventDrawWriteBarrier,
	EventCopyBuffer,
	GpuEvents
};
void NoteGpuEvent(GpuEvent event);
void NoteWriter(bool finished); // Whether a dirty read's last writer had already completed.
void SetFaultPc(uint64_t pc); // The faulting instruction, for the read this thread reports next.
[[nodiscard]] uint64_t FaultPc();
void NoteRead(uint64_t address, bool write, bool gpu_thread, uint64_t queue_ns, uint64_t wait_ns,
              uint64_t copy_ns, uint64_t bytes);
void NoteFault(uint64_t address, bool write, bool gpu_thread, uint64_t buffer_ns,
               uint64_t texture_ns, uint64_t heap_ns);

} // namespace Libs::Graphics::BenchStats

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BENCHSTATS_H_
