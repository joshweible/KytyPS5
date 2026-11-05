#include "graphics/guest_gpu/graphicsRun.h"

#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/guest_gpu/command_processor/pm4Dispatch.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/renderer/benchStats.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/sync.h"
#include "graphics/presentation/videoOut.h"
#include "graphics/presentation/window.h"
#include "graphics/shader/shader.h"
#include "kernel/memory.h"
#include "libs/errno.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <semaphore>
#include <type_traits>
#include <thread>
#include <vector>

namespace Libs::Graphics {

static thread_local Pm4Execution* g_current_execution = nullptr;
static thread_local GuestGpu*     g_gpu_state         = nullptr;
static thread_local GuestGpu*     g_front_state       = nullptr;

namespace {

constexpr uint32_t OpCapacity = 1024;

// Spins through the short waits between packets, then sleeps on the word. The other thread only
// pays for a wake while `sleeping` is set, so that store and its load of it must stay ordered with
// the condition.
template <typename Ready>
void WaitUntil(std::atomic_uint32_t& word, std::atomic_bool& sleeping, Ready ready) {
	for (uint32_t spin = 0; spin < 2048u; spin++) {
		if (ready()) {
			return;
		}
		_mm_pause();
	}
	sleeping.store(true);
	for (;;) {
		const auto seen = word.load();
		if (ready()) {
			break;
		}
		word.wait(seen);
	}
	sleeping.store(false);
}

uint64_t ElapsedNs(std::chrono::steady_clock::time_point from) {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now() - from)
	                                 .count());
}

enum class PacketSide { Front, Back, Sync };

// The registers and the packet stream belong to the front thread. Anything else reaches the
// renderer or guest memory and runs on the GPU thread in packet order. Back packets cannot change
// how the stream is read, so the front thread moves on; every other packet runs while it waits,
// which is how all packets used to run. A new opcode lands here until it is shown to be safe.
PacketSide ClassifyPacket(uint32_t opcode, uint32_t header) {
	switch (opcode) {
		case Pm4::IT_NOP:
			switch (KYTY_PM4_R(header)) {
				case Pm4::R_ZERO:
				case Pm4::R_PUSH_MARKER:
				case Pm4::R_POP_MARKER:
				case Pm4::R_CONTEXT_STATE:
				case Pm4::R_DISPATCH_RESET: return PacketSide::Front;
				case Pm4::R_FLIP:
				case Pm4::R_PSSR: return PacketSide::Back;
				default: return PacketSide::Sync;
			}
		case Pm4::IT_SET_BASE:
		case Pm4::IT_CLEAR_STATE:
		case Pm4::IT_INDEX_BASE:
		case Pm4::IT_INDEX_BUFFER_SIZE:
		case Pm4::IT_INDEX_TYPE:
		case Pm4::IT_SET_CONTEXT_REG:
		case Pm4::IT_SET_SH_REG:
		case Pm4::IT_SET_UCONFIG_REG:
		case Pm4::IT_SET_UCONFIG_REG_INDEX:
		case Pm4::IT_SET_CONTEXT_REG_INDIRECT:
		case Pm4::IT_SET_SH_REG_INDIRECT:
		case Pm4::IT_SET_UCONFIG_REG_INDIRECT: return PacketSide::Front;
		// The 14-dword form is a branch on a guest memory value.
		case Pm4::IT_INDIRECT_BUFFER:
			return KYTY_PM4_LEN(header) == 4u ? PacketSide::Front : PacketSide::Sync;
		case Pm4::IT_DISPATCH_DIRECT:
		case Pm4::IT_DISPATCH_INDIRECT:
		case Pm4::IT_DRAW_INDIRECT:
		case Pm4::IT_DRAW_INDEX_INDIRECT:
		case Pm4::IT_DRAW_INDIRECT_MULTI:
		case Pm4::IT_DRAW_INDEX_INDIRECT_MULTI:
		case Pm4::IT_DRAW_INDEX_2:
		case Pm4::IT_DRAW_INDEX_OFFSET_2:
		case Pm4::IT_DISPATCH_DRAW_PREAMBLE:
		case Pm4::IT_DRAW_INDEX_AUTO:
		case Pm4::IT_NUM_INSTANCES:
		case Pm4::IT_WRITE_DATA:
		case Pm4::IT_EVENT_WRITE:
		case Pm4::IT_RELEASE_MEM: return PacketSide::Back;
		default: return PacketSide::Sync;
	}
}

// The command processor's own writes, which the packets that follow may read.
bool WritesGuestMemory(uint32_t opcode) {
	return opcode == Pm4::IT_WRITE_DATA || opcode == Pm4::IT_RELEASE_MEM;
}

// The front thread reads ahead of execution, as the real command processor's fetcher does: guest
// data read here is not GPU-written unless the game waited for it first, and a wait drains the
// queue. These are the reads that have to see writes by earlier packets.
bool ReadsGuestMemory(uint32_t opcode) {
	return opcode == Pm4::IT_INDIRECT_BUFFER || opcode == Pm4::IT_SET_CONTEXT_REG_INDIRECT ||
	       opcode == Pm4::IT_SET_SH_REG_INDIRECT || opcode == Pm4::IT_SET_UCONFIG_REG_INDIRECT;
}

} // namespace

struct GuestGpu::Op {
	CommandProcessor*            back      = nullptr;
	cp_op_parser_func_t          handler   = nullptr;
	const uint32_t*              buffer    = nullptr;
	uint32_t                     cmd       = 0;
	uint32_t                     dw        = 0;
	uint32_t                     num_dw    = 0;
	bool                         has_state = false;
	Common::UniqueFunction<void> call;
	CpChanges                    changes;
};

// Shader translation runs on this thread and recurses with the shader's structure, so give it the
// same stack on every host.
constexpr size_t GpuThreadStackSize = size_t {64} << 20u;

GuestGpu::GuestGpu(RenderContext& renderer): m_renderer(renderer) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	m_gfx_cp   = std::make_unique<CommandProcessor>(renderer, 0);
	m_gfx_back = std::make_unique<CommandProcessor>(renderer, 0);
	m_gfx_cp->SetBack(m_gfx_back.get());
	m_ops    = std::make_unique<Op[]>(OpCapacity);
	m_thread = std::make_unique<Common::ThreadWithStack>(ThreadRun, this, GpuThreadStackSize);
	m_front_thread =
	    std::make_unique<Common::ThreadWithStack>(FrontRun, this, GpuThreadStackSize);
}

GuestGpu::~GuestGpu() {
	Shutdown();
}

void GuestGpu::Shutdown() {
	std::lock_guard shutdown_lock(m_shutdown_mutex);
	if (m_shutdown_complete) {
		return;
	}
	{
		Common::LockGuard lock(m_queue_mutex);
		m_accepting = false;
		m_stopping  = true;
		m_work_available.SignalAll();
	}
	if (m_front_thread) {
		m_front_thread->Join();
	}
	m_back_stopping.store(true);
	WakeBack();
	if (m_thread) {
		m_thread->Join();
	}
	m_shutdown_complete = true;
}

bool GuestGpu::IsStopping() {
	Common::LockGuard lock(m_queue_mutex);
	return m_stopping;
}

void GuestGpu::SendCommand(Common::UniqueFunction<void>&& command) {
	EXIT_IF(!command);
	if (IsGpuThread()) {
		command();
		return;
	}
	{
		Common::LockGuard lock(m_queue_mutex);
		EXIT_IF(!m_accepting);
		m_commands.push_back(std::move(command));
		m_pending_commands.fetch_add(1, std::memory_order_release);
		m_work_available.Signal();
	}
	if (m_back_sleeping.load()) {
		WakeBack();
	}
}

void GuestGpu::ProcessCommands() {
	while (m_pending_commands.load(std::memory_order_acquire) != 0) {
		Common::UniqueFunction<void> command;
		{
			Common::LockGuard lock(m_queue_mutex);
			EXIT_IF(m_commands.empty());
			command = std::move(m_commands.front());
			m_commands.pop_front();
			EXIT_IF(m_pending_commands.fetch_sub(1, std::memory_order_acq_rel) == 0);
			m_command_running = true;
		}
		const auto from = std::chrono::steady_clock::now();
		command();
		BenchStats::AddGpuThreadTime(BenchStats::GpuCommands, ElapsedNs(from));

		Common::LockGuard lock(m_queue_mutex);
		m_command_running = false;
		if (m_commands.empty() && m_submission_count == 0) {
			m_idle.SignalAll();
		}
	}
}

GuestGpu::Op& GuestGpu::NextOp() {
	const auto write = m_op_write.load(std::memory_order_relaxed);
	if (write - m_front_read >= OpCapacity) {
		m_front_read = m_op_read.load(std::memory_order_acquire);
		// A full queue means the GPU thread is the slower one. Half a queue of room keeps the two
		// from trading a wake for every slot.
		if (write - m_front_read >= OpCapacity) {
			WaitForOp(write - OpCapacity / 2u);
		}
	}
	return m_ops[write % OpCapacity];
}

void GuestGpu::WaitForOp(uint32_t target) {
	const auto reached = [this, target] {
		return static_cast<int32_t>(m_op_read.load() - target) >= 0;
	};
	if (!reached()) {
		const auto from = std::chrono::steady_clock::now();
		m_front_target.store(target, std::memory_order_relaxed);
		WaitUntil(m_op_read, m_front_sleeping, reached);
		BenchStats::AddGpuThreadTime(BenchStats::GpuFrontWait, ElapsedNs(from));
	}
	m_front_read = m_op_read.load(std::memory_order_acquire);
}

void GuestGpu::WakeBack() {
	m_back_wake.fetch_add(1u);
	m_back_wake.notify_one();
}

void GuestGpu::PublishOp() {
	static const bool in_step = std::getenv("KYTY_FRONT_STEP") != nullptr;
	m_op_write.store(m_op_write.load(std::memory_order_relaxed) + 1u);
	if (m_back_sleeping.load()) {
		WakeBack();
	}
	if (in_step) {
		DrainOps();
	}
}

bool GuestGpu::RunNextOp() {
	const auto read = m_op_read.load(std::memory_order_relaxed);
	if (read == m_back_write) {
		m_back_write = m_op_write.load(std::memory_order_acquire);
		if (read == m_back_write) {
			return false;
		}
	}
	auto&      op      = m_ops[read % OpCapacity];
	const bool timed   = BenchStats::Enabled();
	const auto from    = timed ? std::chrono::steady_clock::now()
	                           : std::chrono::steady_clock::time_point {};
	if (op.has_state) {
		op.back->LoadChanges(op.changes);
	}
	uint32_t opcode = 0;
	if (op.handler != nullptr) {
		opcode = (op.cmd >> 8u) & 0xffu;
		EXIT_IF(op.handler(*op.back, op.cmd, op.buffer, op.dw, op.num_dw) + 1u !=
		        KYTY_PM4_LEN(op.cmd));
	} else {
		op.call();
		op.call = Common::UniqueFunction<void> {};
	}
	m_renderer.GetBufferCache().EndPacket();
	if (timed) {
		const auto ns = ElapsedNs(from);
		const auto kind =
		    opcode == Pm4::IT_DISPATCH_DIRECT || opcode == Pm4::IT_DISPATCH_INDIRECT
		        ? BenchStats::GpuDispatch
		    : opcode == Pm4::IT_DRAW_INDEX_2 || opcode == Pm4::IT_DRAW_INDEX_AUTO ||
		            opcode == Pm4::IT_DRAW_INDEX_OFFSET_2 || opcode == Pm4::IT_DRAW_INDIRECT ||
		            opcode == Pm4::IT_DRAW_INDEX_INDIRECT ||
		            opcode == Pm4::IT_DRAW_INDIRECT_MULTI ||
		            opcode == Pm4::IT_DRAW_INDEX_INDIRECT_MULTI
		        ? BenchStats::GpuDraw
		        : BenchStats::GpuOther;
		BenchStats::AddGpuThreadTime(kind, ns);
		BenchStats::NotePacket(opcode, ns);
	}
	m_op_read.store(read + 1u);
	if (m_front_sleeping.load() &&
	    static_cast<int32_t>(read + 1u - m_front_target.load(std::memory_order_relaxed)) >= 0) {
		m_op_read.notify_all();
	}
	return true;
}

void GuestGpu::DrainOps() {
	WaitForOp(m_op_write.load(std::memory_order_relaxed));
	m_unsynced_writes = false;
}

void GuestGpu::PostCall(CommandProcessor* state, Common::UniqueFunction<void>&& call) {
	auto& op     = NextOp();
	op.back      = state != nullptr ? state->Back() : nullptr;
	op.handler   = nullptr;
	op.has_state = state != nullptr;
	op.call      = std::move(call);
	if (state != nullptr) {
		state->SaveChanges(op.changes);
	}
	PublishOp();
}

void GuestGpu::PostPacket(CommandProcessor& back, CommandProcessor* state,
                          cp_op_parser_func_t handler, uint32_t cmd, const uint32_t* buffer,
                          uint32_t dw, uint32_t num_dw, bool writes_memory) {
	auto& op     = NextOp();
	op.back      = &back;
	op.handler   = handler;
	op.buffer    = buffer;
	op.cmd       = cmd;
	op.dw        = dw;
	op.num_dw    = num_dw;
	op.has_state = state != nullptr;
	if (state != nullptr) {
		state->SaveChanges(op.changes);
	}
	m_unsynced_writes = m_unsynced_writes || writes_memory;
	PublishOp();
}

uint32_t GuestGpu::RunPacket(CommandProcessor& cp, cp_op_parser_func_t handler, uint32_t cmd,
                             const uint32_t* buffer, uint32_t dw, uint32_t num_dw) {
	BenchStats::NoteSyncPacket((cmd >> 8u) & 0xffu);
	uint32_t result    = 0;
	auto*    execution = g_current_execution;
	PostCall(nullptr, [&, execution] {
		g_current_execution = execution;
		result              = handler(cp, cmd, buffer, dw, num_dw);
		g_current_execution = nullptr;
	});
	DrainOps();
	return result;
}

void GuestGpu::SendCommandSync(Common::UniqueFunction<void>&& command) {
	EXIT_IF(!command);
	if (IsGpuThread()) {
		command();
		return;
	}
	std::binary_semaphore done {0};
	SendCommand([operation = std::move(command), &done]() mutable {
		operation();
		done.release();
	});
	done.acquire();
}

void GuestGpu::Submit(std::span<const uint32_t> commands) {
	if (commands.empty()) {
		return;
	}
	Submission submission;
	submission.type     = SubmissionType::Commands;
	submission.queue_id = 0;
	submission.commands = commands;
	Enqueue(std::move(submission));
}

void GuestGpu::SubmitCompute(uint32_t queue, std::span<const uint32_t> commands) {
	EXIT_IF(commands.empty());

	EXIT_NOT_IMPLEMENTED(queue < ComputeQueueBase || queue >= ComputeQueueBase + ComputeQueueCount);

	const auto compute_queue = queue - ComputeQueueBase;
	Submission submission;
	submission.type     = SubmissionType::Commands;
	submission.queue_id = 1 + compute_queue;
	submission.commands = commands;
	Enqueue(std::move(submission));
}

void GuestGpu::SubmitFlipPreparation(uint64_t request_id) {
	Submission submission;
	submission.type            = SubmissionType::FlipPreparation;
	submission.queue_id        = 0;
	submission.flip_request_id = request_id;
	Enqueue(std::move(submission));
}

void GuestGpu::SuspendPoint() {
	EXIT_IF(IsGpuThread() || CommandScheduler::InDeferredOperation());
	// Do not hold a queue lock while waiting: asynchronous work may be needed to
	// finish the preceding graphics frame. The first point returns immediately.
	m_suspend_point_ready->acquire();
	Submission submission;
	submission.type = SubmissionType::SuspendPoint;
	Enqueue(std::move(submission));
	m_done_num++;
}

int GuestGpu::GetFrameNum() const {
	return m_done_num;
}

CommandProcessor& GuestGpu::GetProcessor(uint32_t queue_id) {
	if (queue_id == 0) {
		return *m_gfx_cp;
	}
	auto& processor = m_compute_cp[queue_id - 1];
	if (processor == nullptr) {
		const auto event_id          = ComputeQueueBase + queue_id - 1;
		processor                    = std::make_unique<CommandProcessor>(m_renderer, event_id);
		m_compute_back[queue_id - 1] = std::make_unique<CommandProcessor>(m_renderer, event_id);
		processor->SetBack(m_compute_back[queue_id - 1].get());
	}
	return *processor;
}

void CommandProcessor::SaveChanges(CpChanges& changes) {
	static_assert(std::is_trivially_copyable_v<CpState>);
	static_assert(sizeof(CpState) % CpChanges::LineSize == 0);
	const auto* live  = reinterpret_cast<const uint8_t*>(&m_state);
	auto*       sent  = reinterpret_cast<uint8_t*>(&m_sent);
	uint32_t    count = 0;
	for (uint32_t line = 0; line < CpChanges::MaxLines; line++) {
		const auto offset = line * CpChanges::LineSize;
		if (std::memcmp(live + offset, sent + offset, CpChanges::LineSize) != 0) {
			std::memcpy(sent + offset, live + offset, CpChanges::LineSize);
			std::memcpy(changes.data[count].data(), live + offset, CpChanges::LineSize);
			changes.lines[count] = static_cast<uint16_t>(line);
			count++;
		}
	}
	changes.count = count;
}

void CommandProcessor::LoadChanges(const CpChanges& changes) {
	auto* state = reinterpret_cast<uint8_t*>(&m_state);
	for (uint32_t i = 0; i < changes.count; i++) {
		std::memcpy(state + changes.lines[i] * CpChanges::LineSize, changes.data[i].data(),
		            CpChanges::LineSize);
	}
}

void CommandProcessor::Reset() {
	if (m_back != nullptr) {
		g_front_state->PostCall(nullptr, [back = m_back] { back->Reset(); });
	}
	m_state.sh.Reset();
	m_state.ucfg.Reset();
	m_state.ctx.Reset();
	m_saved_ctx.reset();
	m_state.index_type_and_size              = 0;
	m_state.index_buffer_size                = 0;
	m_state.index_base_addr                  = 0;
	m_num_instances                    = 1;
	m_num_instances_addr               = 0;
	m_predicate_skip                   = false;
	m_state.draw_indirect_args_base_addr     = 0;
	m_state.dispatch_indirect_args_base_addr = 0;
}

void CommandProcessor::ApplyContextStateOperation(ContextStateOperation operation) {
	switch (operation) {
		case ContextStateOperation::Clear: m_state.ctx.Reset(); break;
		case ContextStateOperation::Push:
			EXIT_IF(m_saved_ctx.has_value());
			m_saved_ctx.emplace(m_state.ctx);
			break;
		case ContextStateOperation::Pop:
			EXIT_IF(!m_saved_ctx.has_value());
			m_state.ctx = *m_saved_ctx;
			m_saved_ctx.reset();
			break;
		case ContextStateOperation::PushClear:
			EXIT_IF(m_saved_ctx.has_value());
			m_saved_ctx.emplace(m_state.ctx);
			m_state.ctx.Reset();
			break;
	}
}

void CommandProcessor::BufferInit() {
	GetScheduler().Begin(m_state.ctx, m_state.ucfg, m_state.sh);
}

void CommandProcessor::BufferFlush() {
	GetScheduler().Flush();
	// The occlusion counters reach a guest that polls them without waiting for a later fence.
	m_renderer.GetOcclusionQueries().Publish();
}

void CommandProcessor::BufferWait() {
	BufferInit();
	GetScheduler().Finish();
}

void CommandProcessor::WaitForRewind(bool valid) {
	if (!valid) {
		SuspendPm4();
	}
}

bool TestWaitRegMemValue(uint64_t value, uint64_t ref, uint64_t mask, uint32_t func) {
	switch (func) {
		case 0: return true;
		case 1: return (value & mask) < ref;
		case 2: return (value & mask) <= ref;
		case 3: return (value & mask) == ref;
		case 4: return (value & mask) != ref;
		case 5: return (value & mask) >= ref;
		case 6: return (value & mask) > ref;
		default: EXIT("unknown wait compare function: %" PRIu32 "\n", func);
	}

	return false;
}

template <typename T>
void CommandProcessor::WaitRegMem(uint32_t func, const T* addr, T ref, T mask, uint32_t wait_op) {
	EXIT_IF(addr == nullptr);
	if ((wait_op & ~1u) != 0) {
		EXIT("unsupported wait_reg_mem operation: 0x%08" PRIx32 "\n", wait_op);
	}

	if (!TestWaitRegMemValue(*addr, ref, mask, func)) {
		SuspendPm4();
	}
}

template void CommandProcessor::WaitRegMem<uint32_t>(uint32_t, const uint32_t*, uint32_t, uint32_t,
                                                     uint32_t);
template void CommandProcessor::WaitRegMem<uint64_t>(uint32_t, const uint64_t*, uint64_t, uint64_t,
                                                     uint32_t);

void CommandProcessor::MemSemaphore(uint64_t* counter, bool signal, bool write) {
	if (signal) {
		*counter = write ? 1u : *counter + 1u;
	} else if (*counter == 0) {
		SuspendPm4();
	} else {
		(*counter)--;
	}
}

void CommandProcessor::WriteData(uint32_t* dst, const uint32_t* src, uint32_t dw_num,
                                 uint32_t write_control) {
	const uint32_t dst_sel = ((write_control >> 30u) & 0x1u) | ((write_control >> 7u) & 0x1eu);
	const bool     write_one_address = ((write_control >> 16u) & 0x1u) != 0;

	switch (dst_sel) {
		case 0:
		case 2:
		case 4:
		case 5:
		case 6: break;
		default: EXIT("unsupported writeData destination selector 0x%02" PRIx32 "\n", dst_sel);
	}
	if (dw_num == 0) {
		return;
	}

	// Completion labels can be reused immediately; memcpy may write a destination twice.
	auto* output = static_cast<volatile uint32_t*>(dst);
	for (uint32_t i = 0; i < dw_num; i++) {
		*output = src[i];
		output += !write_one_address;
	}
}

void CommandProcessor::WriteReferenceClock(uint64_t dst_address, uint32_t num_bytes) {
	if (dst_address == 0 || (num_bytes != sizeof(uint32_t) && num_bytes != sizeof(uint64_t)) ||
	    (dst_address & (num_bytes - 1u)) != 0) {
		EXIT("invalid reference-clock copy, dst=0x%016" PRIx64 " size=%u\n", dst_address,
		     num_bytes);
	}
	const auto value = Sync::ReadReferenceClock();
	std::memcpy(reinterpret_cast<void*>(dst_address), &value, num_bytes);
	static std::atomic<uint32_t> clock_log_count {0};
	if (clock_log_count.fetch_add(1) < 64) {
		LOGF("\t copy_data reference clock: dst=0x%016" PRIx64 " value=0x%016" PRIx64
		     " size=%u\n",
		     dst_address, value, num_bytes);
	}
}

void CommandProcessor::DmaData(uint8_t dst_sel, uint64_t dst_address_or_offset, uint8_t src_sel,
                               uint64_t src_address_or_offset_or_immediate, uint32_t num_bytes) {
	if (num_bytes == 0 || static_cast<uint32_t>(dst_address_or_offset) == 0x3022cu) {
		return;
	}
	auto decode_gds = [](uint8_t selector, bool& is_gds) {
		switch (selector) {
			case 0:
			case 3: is_gds = false; return true;
			case 1: is_gds = true; return true;
			default: return false;
		}
	};
	if (dst_sel == 2) {
		// kNowhere discards the GL2 prefetch destination without a guest-visible write.
		if (src_sel != 3) {
			EXIT("unsupported dmaData nowhere source selector 0x%02" PRIx8 "\n", src_sel);
		}
		return;
	}
	bool dst_gds = false;
	if (!decode_gds(dst_sel, dst_gds)) {
		EXIT("unsupported dmaData destination selector 0x%02" PRIx8 "\n", dst_sel);
	}
	auto& buffer_cache = m_renderer.GetBufferCache();
	if (src_sel == 2) {
		buffer_cache.FillBuffer(
		    dst_address_or_offset, num_bytes,
		    static_cast<uint32_t>(src_address_or_offset_or_immediate & 0xffffffffu), dst_gds);
		return;
	}
	bool src_gds = false;
	if (!decode_gds(src_sel, src_gds)) {
		EXIT("unsupported dmaData source selector 0x%02" PRIx8 "\n", src_sel);
	}
	if (src_gds && dst_gds) {
		EXIT("unsupported dmaData GDS-to-GDS copy\n");
	}
	buffer_cache.CopyBuffer(dst_address_or_offset, src_address_or_offset_or_immediate, num_bytes,
	                        dst_gds, src_gds);
}

void GuestGpu::Enqueue(Submission submission) {
	Common::LockGuard lock(m_queue_mutex);
	EXIT_IF(!m_accepting);
	m_queues[submission.queue_id].push_back(std::move(submission));
	m_submission_count++;
	m_work_available.Signal();
}

void GuestGpu::WaitForIdle() {
	EXIT_IF(IsGpuThread() || CommandScheduler::InDeferredOperation());
	Common::LockGuard lock(m_queue_mutex);
	while (m_processing || m_command_running || !m_commands.empty() || m_submission_count != 0) {
		m_idle.Wait(&m_queue_mutex);
	}
}

void GuestGpu::ThreadRun(void* data) {
	auto* gpu = static_cast<GuestGpu*>(data);
	EXIT_IF(gpu == nullptr);
	KYTY_PROFILER_THREAD("Thread_Gpu");
	BenchStats::SampleThisThread();
	g_gpu_state = gpu;

	const auto has_work = [gpu] {
		return gpu->m_pending_commands.load() != 0 ||
		       gpu->m_op_read.load(std::memory_order_relaxed) != gpu->m_op_write.load() ||
		       gpu->m_back_stopping.load();
	};
	for (;;) {
		gpu->ProcessCommands();
		if (gpu->RunNextOp()) {
			continue;
		}
		if (gpu->m_back_stopping.load(std::memory_order_acquire) &&
		    gpu->m_pending_commands.load(std::memory_order_acquire) == 0) {
			break;
		}
		const auto idle_from = std::chrono::steady_clock::now();
		WaitUntil(gpu->m_back_wake, gpu->m_back_sleeping, has_work);
		BenchStats::AddGpuThreadTime(BenchStats::GpuIdle, ElapsedNs(idle_from));
	}
	gpu->m_gfx_back->BufferWait();
	g_gpu_state = nullptr;
}

void GuestGpu::FrontRun(void* data) {
	auto* gpu = static_cast<GuestGpu*>(data);
	EXIT_IF(gpu == nullptr);
	KYTY_PROFILER_THREAD("Thread_GpuFront");
	g_front_state = gpu;

	for (;;) {
		bool has_submission = false;
		{
			Common::LockGuard lock(gpu->m_queue_mutex);
			has_submission = gpu->m_submission_count != 0;
		}
		// Idle means the GPU thread has finished too.
		if (!has_submission) {
			gpu->DrainOps();
		}

		Submission submission;
		bool       should_stop = false;
		{
			Common::LockGuard lock(gpu->m_queue_mutex);
			while (gpu->m_submission_count == 0 && !gpu->m_stopping) {
				gpu->m_processing = false;
				gpu->m_idle.SignalAll();
				gpu->m_work_available.Wait(&gpu->m_queue_mutex);
			}
			if (gpu->m_submission_count == 0) {
				gpu->m_processing = false;
				gpu->m_idle.SignalAll();
				should_stop = true;
			} else {
				int selected_queue = -1;
				for (uint32_t offset = 0; offset < QueueCount; offset++) {
					const auto id = (gpu->m_next_queue + offset) % QueueCount;
					if (!gpu->m_queues[id].empty() && !gpu->m_queues[id].front().blocked) {
						selected_queue = static_cast<int>(id);
						break;
					}
				}
				if (selected_queue < 0) {
					gpu->m_processing = false;
					gpu->m_work_available.WaitFor(&gpu->m_queue_mutex, 100);
					for (auto& queue: gpu->m_queues) {
						if (!queue.empty()) {
							queue.front().blocked = false;
						}
					}
					continue;
				}
				auto& queue = gpu->m_queues[static_cast<uint32_t>(selected_queue)];
				submission  = std::move(queue.front());
				queue.pop_front();
				gpu->m_submission_count--;
				gpu->m_next_queue = (static_cast<uint32_t>(selected_queue) + 1) % QueueCount;
				gpu->m_processing = true;
			}
		}
		if (should_stop) {
			g_front_state = nullptr;
			return;
		}

		const auto from     = std::chrono::steady_clock::now();
		const bool complete = gpu->Process(submission);
		BenchStats::AddGpuThreadTime(BenchStats::GpuFrontActive, ElapsedNs(from));

		Common::LockGuard lock(gpu->m_queue_mutex);
		if (!complete) {
			submission.blocked = true;
			gpu->m_queues[submission.queue_id].push_front(std::move(submission));
			gpu->m_submission_count++;
		} else {
			for (auto& queue: gpu->m_queues) {
				if (!queue.empty()) {
					queue.front().blocked = false;
				}
			}
		}
	}
}

bool GuestGpu::Process(Submission& submission) {
	auto& cp   = GetProcessor(submission.queue_id);
	auto* back = cp.Back();

	PostCall(nullptr, [back] { back->BufferInit(); });
	bool complete = true;

	switch (submission.type) {
		case SubmissionType::Commands:
			complete =
			    cp.Process(submission.execution, submission.commands) == Pm4ProcessResult::Complete;
			if (submission.execution.MadeProgress()) {
				PostCall(nullptr, [back] { back->BufferFlush(); });
			}
			break;
		case SubmissionType::FlipPreparation:
			PostCall(nullptr, [back, request_id = submission.flip_request_id] {
				back->PrepareCpuFlip(request_id);
			});
			break;
		case SubmissionType::SuspendPoint:
			PostCall(nullptr, [this, back] {
				// Age caches at the frame boundary, not once per command list.
				m_renderer.RunGarbageCollector();
				m_renderer.GetCommandScheduler().DeferPriorityOperation(
				    [ready = m_suspend_point_ready] { ready->release(); });
				back->BufferFlush();
			});
			cp.Reset();
			break;
	}

	return complete;
}

Pm4ProcessResult CommandProcessor::Process(Pm4Execution&             execution,
                                           std::span<const uint32_t> commands) {
	KYTY_PROFILER_BLOCK("CommandProcessor::Process");
	EXIT_IF(g_current_execution != nullptr);
	EXIT_IF(commands.size() > UINT32_MAX);
	if (execution.m_buffer_stack.empty() && !commands.empty()) {
		execution.m_buffer_stack.push_back({commands});
	}
	execution.m_suspended     = false;
	execution.m_made_progress = false;

	struct ExecutionScope {
		explicit ExecutionScope(Pm4Execution& execution) { g_current_execution = &execution; }
		~ExecutionScope() { g_current_execution = nullptr; }
	} execution_scope(execution);

	if (static const bool lookahead = std::getenv("KYTY_DISPATCH_LOOKAHEAD") == nullptr ||
	                                  std::getenv("KYTY_DISPATCH_LOOKAHEAD")[0] != '0';
	    lookahead && execution.m_buffer_stack.size() == 1u &&
	    execution.m_buffer_stack.back().offset_dw == 0u) {
		g_front_state->PostCall(this, [back = m_back, commands = execution.m_buffer_stack.back()
		                                                             .commands] {
			back->DispatchLookahead(commands);
		});
	}

	ProcessPm4(execution);
	return execution.m_buffer_stack.empty() ? Pm4ProcessResult::Complete
	                                        : Pm4ProcessResult::Blocked;
}

void CommandProcessor::ProcessIndirectBuffer(std::span<const uint32_t> commands, bool chain) {
	EXIT_IF(g_current_execution == nullptr);
	EXIT_IF(!g_current_execution->m_next_buffer.empty());
	g_current_execution->m_next_buffer = commands;
	g_current_execution->m_chain       = chain;
}

void CommandProcessor::SuspendPm4() {
	EXIT_IF(g_current_execution == nullptr);
	g_current_execution->m_suspended = true;
}

// Only after a pipeline was built: ordinary frames never pay for a walk.
void CommandProcessor::DispatchLookahead(std::span<const uint32_t> commands) {
	const auto created = m_renderer.GetPipelineCache().ComputePipelinesCreated();
	if (created != m_lookahead_created) {
		m_lookahead_created = created;
		RunDispatchLookahead(commands);
		m_lookahead_created = m_renderer.GetPipelineCache().ComputePipelinesCreated();
	}
}

// Walks the packets of this submission ahead of their execution: a second processor takes the
// shader register writes, and each dispatch it reaches starts its pipeline on a worker, so a burst
// of new pipelines compiles in parallel instead of one at a time on this thread.
void CommandProcessor::RunDispatchLookahead(std::span<const uint32_t> commands) {
	KYTY_PROFILER_FUNCTION();
	constexpr uint32_t MaxDispatches = 256;
	constexpr uint32_t MaxPackets    = 1u << 16u;

	auto shadow      = std::make_unique<CommandProcessor>(m_renderer, m_interrupt_event_id);
	shadow->m_state.sh = m_state.sh;

	auto&    pipelines  = m_renderer.GetPipelineCache();
	std::vector<Pm4Execution::BufferCursor> stack {{commands}};
	uint32_t dispatches = 0;
	for (uint32_t packets = 0; !stack.empty() && dispatches < MaxDispatches && packets < MaxPackets;
	     packets++) {
		auto& cursor = stack.back();
		if (cursor.offset_dw >= cursor.commands.size()) {
			stack.pop_back();
			continue;
		}
		const auto* packet    = cursor.commands.data() + cursor.offset_dw;
		const auto  total_dw  = static_cast<uint32_t>(cursor.commands.size());
		const auto  remaining = total_dw - cursor.offset_dw;
		const auto  header    = packet[0];
		if (header == 0x80000000u || Pm4::PacketSizeDw(header) == 1u) {
			cursor.offset_dw++;
			continue;
		}
		const auto packet_dw = KYTY_PM4_LEN(header);
		if ((header >> 30u) != 3u || remaining < 2u || packet_dw > remaining || packet_dw == 0u) {
			break;
		}
		const auto opcode = (header >> 8u) & 0xffu;
		if (opcode == Pm4::IT_SET_SH_REG) {
			if (g_cp_op_func[opcode](*shadow, header & ~1u, packet + 1, remaining, total_dw) + 1u !=
			    packet_dw) {
				break;
			}
		} else if (opcode == Pm4::IT_INDIRECT_BUFFER) {
			// Calls and chains are followed; branches (the 14-dword form) end the walk.
			if (packet_dw != 4u) {
				break;
			}
			const auto* address = reinterpret_cast<const uint32_t*>(
			    packet[1] | (static_cast<uint64_t>(packet[2]) << 32u));
			const auto size  = packet[3] & 0xfffffu;
			const bool chain = (packet[3] & (1u << 20u)) != 0u;
			cursor.offset_dw += packet_dw;
			if (size == 0u) {
				continue;
			}
			if (address == nullptr) {
				break;
			}
			const std::span<const uint32_t> target(address, size);
			if (chain) {
				stack.back() = {target};
			} else {
				stack.push_back({target});
			}
			continue;
		} else if (opcode == Pm4::IT_DISPATCH_DIRECT) {
			dispatches++;
			if (packet_dw == 5u && packet[1] != 0u && packet[2] != 0u && packet[3] != 0u) {
				const std::array<uint32_t, 3> groups {packet[1], packet[2], packet[3]};
				pipelines.PrefetchComputePipeline(shadow->m_state.sh.GetCs(), packet[4], &groups);
			}
		} else if (opcode == Pm4::IT_DISPATCH_INDIRECT) {
			dispatches++;
			if ((header & ~1u) == 0xc0021600u) {
				pipelines.PrefetchComputePipeline(shadow->m_state.sh.GetCs(), packet[3], nullptr);
			} else if ((header & ~1u) == 0xc0011600u) {
				pipelines.PrefetchComputePipeline(shadow->m_state.sh.GetCs(), packet[2], nullptr);
			}
		}
		cursor.offset_dw += packet_dw;
	}
}

void CommandProcessor::ProcessPm4(Pm4Execution& execution) {
	const bool debug_dump = graphics_debug_dump_enabled();
	while (!execution.m_buffer_stack.empty()) {
		auto& cursor = execution.m_buffer_stack.back();
		if (cursor.offset_dw == cursor.commands.size()) {
			execution.m_buffer_stack.pop_back();
			continue;
		}

		const auto* const packet        = cursor.commands.data() + cursor.offset_dw;
		const auto        total_dw      = static_cast<uint32_t>(cursor.commands.size());
		const auto        remaining_dw  = total_dw - cursor.offset_dw;
		const auto        packet_header = packet[0];
		const auto        opcode        = (packet_header >> 8u) & 0xffu;

		if (packet_header == 0x80000000u || Pm4::PacketSizeDw(packet_header) == 1u) {
			cursor.offset_dw++;
			execution.m_made_progress = true;
			continue;
		}

		EXIT_NOT_IMPLEMENTED(remaining_dw < 2);

		if (debug_dump) {
			LOGF("CP packet: offset=0x%05" PRIx32 " cmd_id=0x%08" PRIx32 " op=0x%02" PRIx32
			     " len=%" PRIu32 "\n",
			     cursor.offset_dw, packet_header, opcode, KYTY_PM4_LEN(packet_header));
		}

		if ((packet_header & 1u) != 0 && ShouldSkipPredicatedPackets()) {
			auto packet_dw = KYTY_PM4_LEN(packet_header);
			EXIT_NOT_IMPLEMENTED(packet_dw > remaining_dw);
			cursor.offset_dw += packet_dw;
			execution.m_made_progress = true;
			continue;
		}

		auto handler = g_cp_op_func[opcode];

		if (handler == nullptr) {
			const auto offset = cursor.offset_dw;
			LOGF("unknown PM4 packet: data=0x%016" PRIx64 ", num_dw=%" PRIu32
			     ", offset=0x%05" PRIx32 ", current=0x%016" PRIx64 "\n",
			     reinterpret_cast<uint64_t>(packet - offset), total_dw, offset,
			     reinterpret_cast<uint64_t>(packet));
			const auto  dump_begin = (offset > 8 ? offset - 8 : 0);
			const auto  dump_end   = std::min<uint32_t>(total_dw, offset + 16);
			auto* const base       = packet - offset;
			for (uint32_t i = dump_begin; i < dump_end; i++) {
				LOGF("\t%05" PRIx32 "%s %08" PRIx32 "\n", i, (i == offset ? ":" : " "), base[i]);
			}
			EXIT("unknown op\n\t%05" PRIx32 ":\n\tcmd_id = %08" PRIx32 "\n", offset, packet_header);
		}

		const auto cmd  = packet_header & ~1u;
		auto       side = ClassifyPacket(opcode, packet_header);
		// A write into the packets still to come has to land before they are read.
		if (opcode == Pm4::IT_WRITE_DATA) {
			const auto dst = packet[2] | (static_cast<uint64_t>(packet[3]) << 32u);
			for (const auto& entry: execution.m_buffer_stack) {
				const auto first = reinterpret_cast<uint64_t>(entry.commands.data());
				if (dst + (uint64_t {KYTY_PM4_LEN(packet_header)} * 4u) > first &&
				    dst < first + entry.commands.size_bytes()) {
					side = PacketSide::Sync;
				}
			}
		}
		uint32_t packet_dw = 0;
		switch (side) {
			case PacketSide::Front:
				if (ReadsGuestMemory(opcode)) {
					g_front_state->SyncWrites();
				}
				packet_dw = handler(*this, cmd, packet + 1, remaining_dw, total_dw) + 1;
				break;
			case PacketSide::Back:
				packet_dw = KYTY_PM4_LEN(packet_header);
				g_front_state->PostPacket(*m_back, opcode == Pm4::IT_NUM_INSTANCES ? nullptr : this,
				                          handler, cmd, packet + 1, remaining_dw, total_dw,
				                          WritesGuestMemory(opcode));
				break;
			case PacketSide::Sync:
				packet_dw = g_front_state->RunPacket(*this, handler, cmd, packet + 1, remaining_dw,
				                                     total_dw) +
				            1;
				break;
		}
		EXIT_IF(packet_dw > remaining_dw);
		if (execution.m_suspended) {
			return;
		}
		cursor.offset_dw += packet_dw;
		execution.m_made_progress = true;
		if (!execution.m_next_buffer.empty()) {
			// Chains and taken branches reuse the fetcher; only calls retain a return cursor.
			if (execution.m_chain) {
				cursor = {execution.m_next_buffer};
			} else {
				execution.m_buffer_stack.push_back({execution.m_next_buffer});
			}
			execution.m_next_buffer = {};
		}
	}
}

void CommandProcessor::SetIndexType(uint32_t index_type_and_size) {
	m_state.index_type_and_size = index_type_and_size & 0x3u;
}

void CommandProcessor::SetIndexBaseAddress(uint64_t index_base_addr) {
	m_state.index_base_addr = index_base_addr;
}

void CommandProcessor::SetIndexBufferSize(uint32_t index_buffer_size) {
	m_state.index_buffer_size = index_buffer_size;
}

void CommandProcessor::SetDrawIndirectArgsBaseAddress(uint64_t draw_indirect_args_base_addr) {
	m_state.draw_indirect_args_base_addr = draw_indirect_args_base_addr;
}

void CommandProcessor::SetDispatchIndirectArgsBaseAddress(
    uint64_t dispatch_indirect_args_base_addr) {
	m_state.dispatch_indirect_args_base_addr = dispatch_indirect_args_base_addr;
}

void CommandProcessor::SetNumInstances(uint32_t num_instances) {
	if (num_instances == 0) {
		num_instances = 1;
	}

	m_num_instances      = num_instances;
	m_num_instances_addr = 0;
}

uint32_t CommandProcessor::NumInstances() {
	if (m_num_instances_addr != 0) {
		m_num_instances      = *reinterpret_cast<const uint32_t*>(m_num_instances_addr);
		m_num_instances_addr = 0;
	}
	return m_num_instances;
}

void CommandProcessor::SetPredication(uint32_t condition, uint32_t op, uint32_t wait_op,
                                      const volatile void* address) {
	uint64_t value = 0;

	switch (op) {
		case 0x00:
			m_predicate_skip = false;
			return;
		case 0x01: {
			EXIT_NOT_IMPLEMENTED(address == nullptr);
			m_renderer.GetOcclusionQueries().Publish();
			// One begin/end pair per DB; bit 63 marks each counter ready.
			constexpr uint64_t ready_bit = 1ull << 63u;
			const auto* results = reinterpret_cast<const volatile uint64_t*>(address);
			for (uint32_t db = 0; db < 16u; db++) {
				const auto begin = results[db * 2u];
				const auto end   = results[db * 2u + 1u];
				if ((begin & end & ready_bit) == 0) {
					if (wait_op == 0) {
						SuspendPm4();
					} else {
						m_predicate_skip = false;
					}
					return;
				}
				value += end - begin;
			}
		} break;
		case 0x03:
			// The wait selector applies only to Z-pass query readiness.
			EXIT_NOT_IMPLEMENTED(address == nullptr);
			value = *reinterpret_cast<const volatile uint64_t*>(address);
			break;
		default: EXIT("unknown predication op: 0x%08" PRIx32 "\n", op);
	}
	switch (condition) {
		case 0x00: m_predicate_skip = (value != 0); break;
		case 0x01: m_predicate_skip = (value == 0); break;
		default: EXIT("unknown predication condition: 0x%08" PRIx32 "\n", condition);
	}
}

void CommandProcessor::DrawIndex(DrawIndexArgs args) {
	args.index_type_and_size = m_state.index_type_and_size;
	if (args.instance_count == 0) {
		args.instance_count = NumInstances();
	}
	m_renderer.GetRenderExecutor().DrawIndex(CurrentBuffer(), args);
}

const void* CommandProcessor::IndexAddress(uint32_t first_index) const {
	uint64_t index_size = 0;
	switch (m_state.index_type_and_size) {
		case 0: index_size = 2; break;
		case 1: index_size = 4; break;
		case 2: index_size = 1; break;
		default: EXIT("unknown index_type_and_size: %u\n", m_state.index_type_and_size);
	}
	return reinterpret_cast<const void*>(m_state.index_base_addr + first_index * index_size);
}

void CommandProcessor::DrawIndexOffset(uint32_t index_offset, uint32_t index_count) {
	DrawIndex({.index_count = index_count, .index_addr = IndexAddress(index_offset)});
}

static void PatchIndirectDrawOffsets(CommandProcessor& cp, IndirectDrawRegisters registers,
                                     uint32_t vertex_offset, uint32_t instance_offset,
                                     uint32_t first_index, uint32_t draw_index) {
	const auto write = [&](uint32_t location, uint32_t value) {
		if (location == Pm4::SH_NOP) {
			return;
		}
		EXIT_NOT_IMPLEMENTED(location >= Pm4::SH_NUM ||
		                     g_hw_sh_indirect_func[location] == nullptr);
		g_hw_sh_indirect_func[location](cp, location, value);
	};
	write(registers.vertex_offset, vertex_offset);
	write(registers.instance_offset, instance_offset);
	write(registers.index_offset, first_index);
	write(registers.draw_index, draw_index);
}

// Writes the GS user-data slots the packet patches; false when the shaders cannot read the
// offsets from GPU records: a first-index or subdraw-index destination, or another stage's
// user data.
static bool IndirectDrawOffsets(IndirectDrawRegisters registers, ShaderIndirectOffsets& offsets) {
	if (registers.index_offset != Pm4::SH_NOP || registers.draw_index != Pm4::SH_NOP) {
		return false;
	}
	for (const auto [location, slot]: {std::pair {registers.vertex_offset, &offsets.vertex},
	                                   std::pair {registers.instance_offset, &offsets.instance}}) {
		if (location == Pm4::SH_NOP) {
			continue;
		}
		const uint32_t index = location - Pm4::SPI_SHADER_USER_DATA_GS_0;
		if (index >= HW::UserSgprInfo::SGPRS_MAX) {
			return false;
		}
		*slot = static_cast<uint8_t>(index);
	}
	return true;
}

void CommandProcessor::DrawIndirect(uint32_t data_offset, IndirectDrawRegisters registers,
                                    uint32_t draw_initiator, bool indexed) {
	const auto args_size =
	    indexed ? sizeof(vk::DrawIndexedIndirectCommand) : sizeof(vk::DrawIndirectCommand);
	DrawIndirectMulti(data_offset, 1, nullptr, args_size, registers, draw_initiator, indexed);
}

void CommandProcessor::DrawIndirectMulti(uint32_t data_offset, uint32_t max_count_or_count,
                                         const volatile uint32_t* count_addr,
                                         uint32_t stride_in_bytes, IndirectDrawRegisters registers,
                                         uint32_t draw_initiator, bool indexed) {
	EXIT_NOT_IMPLEMENTED((draw_initiator & ~0x20u) != (indexed ? 0u : 2u));
	EXIT_NOT_IMPLEMENTED(m_state.draw_indirect_args_base_addr == 0);
	// The guest records match Vulkan's indirect commands byte for byte.
	EXIT_NOT_IMPLEMENTED((stride_in_bytes & 3u) != 0 ||
	                     stride_in_bytes < (indexed ? sizeof(vk::DrawIndexedIndirectCommand)
	                                                : sizeof(vk::DrawIndirectCommand)));

	const auto draw_count = count_addr == nullptr
	                            ? max_count_or_count
	                            : std::min(uint32_t {*count_addr}, max_count_or_count);
	if (draw_count == 0) {
		return;
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::IndirectDraws, draw_count);

	auto&      executor  = m_renderer.GetRenderExecutor();
	const auto args_addr = m_state.draw_indirect_args_base_addr + data_offset;
	// NUM_INSTANCES keeps the last draw's instance count.
	m_num_instances_addr = args_addr + static_cast<uint64_t>(draw_count - 1) * stride_in_bytes +
	                       offsetof(vk::DrawIndirectCommand, instanceCount);
	// The GPU reads the records if the shaders allow; the VS prolog removes Vulkan's bases
	// from the index VGPRs and copies them only to the user SGPRs the packet patches.
	DrawIndirectArgs records {args_addr, draw_count, stride_in_bytes};
	if (IndirectDrawOffsets(registers, records.offsets)) {
		if (!indexed && executor.DrawAuto(CurrentBuffer(), {.indirect = records})) {
			return;
		}
		if (indexed && m_state.index_buffer_size != 0 &&
		    executor.DrawIndex(CurrentBuffer(), {.index_count         = m_state.index_buffer_size,
		                                         .index_addr          = IndexAddress(0),
		                                         .index_type_and_size = m_state.index_type_and_size,
		                                         .indirect            = records})) {
			return;
		}
	}

	const uint32_t max_indices = m_state.index_buffer_size != 0 ? m_state.index_buffer_size : UINT32_MAX;
	for (uint32_t i = 0; i < draw_count; i++) {
		const auto addr = args_addr + static_cast<uint64_t>(i) * stride_in_bytes;
		if (!indexed) {
			const auto& args = *reinterpret_cast<const vk::DrawIndirectCommand*>(addr);
			PatchIndirectDrawOffsets(*this, registers, args.firstVertex, args.firstInstance, 0, i);
			executor.DrawAuto(CurrentBuffer(), {.vertex_count   = args.vertexCount,
			                                    .instance_count = args.instanceCount,
			                                    .first_vertex   = args.firstVertex,
			                                    .first_instance = args.firstInstance,
			                                    .offset_source  = DrawOffsetSource::IndirectArgs});
			continue;
		}

		const auto& args = *reinterpret_cast<const vk::DrawIndexedIndirectCommand*>(addr);
		PatchIndirectDrawOffsets(*this, registers, static_cast<uint32_t>(args.vertexOffset),
		                         args.firstInstance, args.firstIndex, i);
		executor.DrawIndex(CurrentBuffer(), {.index_count = std::min(args.indexCount, max_indices),
		                                     .index_addr  = IndexAddress(args.firstIndex),
		                                     .instance_count      = args.instanceCount,
		                                     .index_type_and_size = m_state.index_type_and_size,
		                                     .base_vertex         = args.vertexOffset,
		                                     .first_instance      = args.firstInstance,
		                                     .offset_source = DrawOffsetSource::IndirectArgs});
	}
}

void CommandProcessor::DispatchDirect(uint32_t thread_group_x, uint32_t thread_group_y,
                                      uint32_t thread_group_z, uint32_t mode) {
	m_state.sh.SetCsWaveSize(Pm4::ComputeWaveSize(mode));
	m_renderer.GetRenderExecutor().DispatchDirect(CurrentBuffer(), thread_group_x, thread_group_y,
	                                              thread_group_z, mode, IsAsyncComputeQueue());
}

void CommandProcessor::ProcessPssr(const Pssr::Request& request) {
	EXIT_NOT_IMPLEMENTED(IsAsyncComputeQueue());
	m_renderer.GetRenderExecutor().ProcessPssr(CurrentBuffer(), request);
}

void CommandProcessor::DispatchIndirect(uint64_t args_addr, uint32_t mode) {
	EXIT_NOT_IMPLEMENTED(args_addr == 0 || (args_addr & 3u) != 0);
	if ((mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0) {
		const auto* args = reinterpret_cast<const vk::DispatchIndirectCommand*>(args_addr);
		DispatchDirect(args->x, args->y, args->z, mode);
		return;
	}
	m_state.sh.SetCsWaveSize(Pm4::ComputeWaveSize(mode));
	m_renderer.GetRenderExecutor().DispatchIndirect(CurrentBuffer(), args_addr, mode,
	                                                IsAsyncComputeQueue());
}

void CommandProcessor::DrawIndexAuto(DrawAutoArgs args) {
	if (args.instance_count == 0) {
		args.instance_count = NumInstances();
	}
	m_renderer.GetRenderExecutor().DrawAuto(CurrentBuffer(), args);
}

void CommandProcessor::WaitFlipDone(uint32_t video_out_handle, uint32_t display_buffer_index) {
	BufferFlush();

	m_renderer.GetVideoOut().WaitFlipDone(static_cast<int>(video_out_handle),
	                                      static_cast<int>(display_buffer_index));
}

void CommandProcessor::ReleaseMem(uint32_t data_sel, uint32_t interrupt_selector, void* address,
                                  uint64_t data, uint32_t interrupt_context_id) {
	if (graphics_debug_dump_enabled()) {
		LOGF("CommandProcessor::ReleaseMem()\n"
		     "\t data_sel           = %" PRIu32 "\n"
		     "\t interrupt_selector = %" PRIu32 "\n"
		     "\t address            = 0x%016" PRIx64 "\n"
		     "\t data               = 0x%016" PRIx64 "\n"
		     "\t interrupt_context  = 0x%08" PRIx32 "\n",
		     data_sel, interrupt_selector, reinterpret_cast<uint64_t>(address), data,
		     interrupt_context_id);
	}

	// QueuedInterrupt: kNone, kOnly, kAfterWriteConfirm, kDataAfterWriteConfirm, kSendIntCtx.
	EXIT_NOT_IMPLEMENTED(interrupt_selector > 4);
	const bool interrupt = (interrupt_selector != 0 && interrupt_selector != 3);
	// kSendIntCtx has no label, and kOnly writes none on graphics queues.
	if (address == nullptr || interrupt_selector == 4 ||
	    (interrupt_selector == 1 && !IsAsyncComputeQueue())) {
		data_sel = 0;
	}
	if (data_sel == 0 && !interrupt) {
		return;
	}

	// Dumped occlusion counters are written before any end-of-pipe signal reaches the guest.
	m_renderer.GetOcclusionQueries().Publish();

	// A label beside GPU-written bytes goes to the backing store, so they are not downloaded.
	// Other labels go through the guest mapping: its fault keeps a clean cached copy current and
	// downloads GPU-written bytes under the label before the store.
	const auto write_label = [this, address](const void* value, uint64_t size) {
		const auto vaddr   = reinterpret_cast<uint64_t>(address);
		auto&      buffers = m_renderer.GetBufferCache();
		if (!GuestRange {vaddr, size}.Valid() || !buffers.IsRegionGpuModified(vaddr, size) ||
		    buffers.HasGpuDirtyBytes(vaddr, size) ||
		    !LibKernel::Memory::TryWriteBacking(vaddr, value, size)) {
			std::memcpy(address, value, size);
		}
	};
	// EndOfPipeData: kNone, kImmediate32b, kImmediate64b, kRefClk; 5 reads GDS.
	switch (data_sel) {
		case 0: break;
		case 1: {
			const auto value = static_cast<uint32_t>(data);
			write_label(&value, sizeof(value));
			break;
		}
		case 3: data = Sync::ReadReferenceClock(); [[fallthrough]];
		case 2: write_label(&data, sizeof(data)); break;
		case 5: {
			GetScheduler().Finish();
			const auto gds = static_cast<uint32_t>(data);
			Sync::ReadGds(*m_renderer.GetBufferCache().GetGdsBuffer(),
			              static_cast<uint32_t*>(address), gds & 0xffffu, gds >> 16u);
			break;
		}
		default: EXIT("unknown release_mem data selector: %" PRIu32 "\n", data_sel);
	}

	// GuestGpu::Process's end-of-slice flush submits the work this interrupt waits on.
	if (interrupt) {
		GetScheduler().DeferPriorityOperation(
		    [&renderer = m_renderer, event_id = m_interrupt_event_id, interrupt_context_id] {
			    renderer.TriggerInterrupt(event_id, interrupt_context_id);
		    });
	}
}

void CommandProcessor::TriggerEvent(uint32_t event_type, uint32_t event_index,
                                    uint64_t event_address) {
	if (graphics_debug_dump_enabled()) {
		LOGF("CommandProcessor::TriggerEvent()\n"
		     "\t event_type  = 0x%08" PRIx32 "\n"
		     "\t event_index = 0x%08" PRIx32 "\n"
		     "\t address     = 0x%016" PRIx64 "\n",
		     event_type, event_index, event_address);
	}

	const auto valid_cache_event_index = event_index == 0x00000000 || event_index == 0x00000007;
	// Partial flushes and cache writebacks only order GPU work, which the renderer's resource
	// barriers already order.
	switch (event_type) {
		// CsPartialFlush, GsPartialFlush, PsPartialFlush.
		case 0x00000007:
		case 0x0000000f:
		case 0x00000010: break;
		// CbDbDataWritebackInvalidate, CbDataWritebackInvalidate, DbDataWritebackInvalidate,
		// DbMetadataWritebackInvalidate, CbMetadataWritebackInvalidate.
		case 0x00000016:
		case 0x00000031:
		case 0x0000002a:
		case 0x0000002c:
		case 0x0000002e:
			if (!valid_cache_event_index) {
				EXIT("unknown event type: 0x%08" PRIx32 ", 0x%08" PRIx32 "\n", event_type,
				     event_index);
			}
			break;
		case 0x0000000d:
		case 0x0000000e:
		case 0x00000012:
		case 0x00000017:
		case 0x00000018:
		case 0x00000019:
		case 0x0000001a:
		case 0x0000001b:
		case 0x00000038:
		case 0x0000003a:
			LOGF("\t temporary: ignoring unsupported event_write type 0x%08" PRIx32
			     ", index 0x%08" PRIx32 "\n",
			     event_type, event_index);
			break;
		case 0x00000039: {
			if (event_index != 0x00000001 || event_address == 0 || (event_address & 0x7u) != 0) {
				EXIT("invalid occlusion-counter dump: index=0x%08" PRIx32 ", address=0x%016" PRIx64
				     "\n",
				     event_index, event_address);
			}
			if (auto& queries = m_renderer.GetOcclusionQueries(); queries.Enabled()) {
				queries.Dump(event_address);
				break;
			}
			static std::once_flag warning_once;
			std::call_once(warning_once, [] {
				std::printf("Warning: game uses occlusion queries, which are treated as always "
				            "visible without --occlusion-queries; GPU usage may be higher and FPS "
				            "may be lower.\n");
			});

			// Without host occlusion queries, publish an always-visible result. The PS5 layout
			// contains one interleaved begin/end pair per DB, and bit 63 marks a result ready.
			constexpr uint64_t ready_bit    = 1ull << 63u;
			constexpr uint64_t counter_mask = ready_bit - 1u;
			auto*              results      = reinterpret_cast<volatile uint64_t*>(event_address);
			const auto         value        = ready_bit | m_synthetic_occlusion_counter;
			for (uint32_t db = 0; db < 16u; db++) {
				results[db * 2u] = value;
			}
			m_synthetic_occlusion_counter = (m_synthetic_occlusion_counter + 1u) & counter_mask;
			break;
		}
		default:
			EXIT("unknown event type: 0x%08" PRIx32 ", 0x%08" PRIx32 "\n", event_type, event_index);
	}
}

void CommandProcessor::Flip(int handle, int index, int flip_mode, int64_t flip_arg) {
	if (graphics_debug_dump_enabled()) {
		LOGF("CommandProcessor::Flip()\n");
	}
	m_renderer.GetOcclusionQueries().Publish();

	auto& command = CurrentBuffer();
	auto  request = Sync::PrepareVideoOutFlip(command, handle, index, flip_mode, flip_arg);
	GetScheduler().DeferPriorityOperation(
	    [&renderer = m_renderer, request] { renderer.GetVideoOut().CompleteFlip(request); });
	GetScheduler().Flush();
}

void CommandProcessor::PrepareCpuFlip(uint64_t request_id) {
	auto& command = CurrentBuffer();
	m_renderer.GetVideoOut().PrepareFlip(request_id, command);
	GetScheduler().DeferPriorityOperation(
	    [this, request_id] { m_renderer.GetVideoOut().CompleteFlip(request_id); });
	GetScheduler().Flush();
}

bool GuestGpu::IsGpuThread() noexcept {
	return g_gpu_state != nullptr;
}

} // namespace Libs::Graphics
