#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/benchStats.h"

#include "common/assert.h"
#include "common/debugCounters.h"
#include "graphics/host_gpu/graphicContext.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <optional>
#include <vector>

namespace Libs::Graphics {

static thread_local CommandScheduler* g_deferred_callback_scheduler = nullptr;

CommandScheduler::CommandPool::CommandPool(GraphicContext& graphics, MasterSemaphore& master)
    : m_graphics(graphics), m_master(master) {
	EXIT_IF(graphics.queue_family == static_cast<uint32_t>(-1));
	vk::CommandPoolCreateInfo create {};
	create.queueFamilyIndex = graphics.queue_family;
	create.flags            = vk::CommandPoolCreateFlagBits::eTransient |
	                          vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
	const auto result       = graphics.device.createCommandPool(&create, nullptr, &m_pool);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_pool == nullptr);
}

CommandScheduler::CommandPool::~CommandPool() {
	m_graphics.device.destroyCommandPool(m_pool, nullptr);
}

size_t CommandScheduler::CommandPool::Grow() {
	const auto first = m_ticks.size();
	m_ticks.resize(first + GrowStep);
	m_buffers.resize(first + GrowStep);

	vk::CommandBufferAllocateInfo allocate {};
	allocate.commandPool        = m_pool;
	allocate.level              = vk::CommandBufferLevel::ePrimary;
	allocate.commandBufferCount = static_cast<uint32_t>(GrowStep);
	EXIT_IF(m_graphics.device.allocateCommandBuffers(&allocate, m_buffers.data() + first) !=
	        vk::Result::eSuccess);
	return first;
}

vk::CommandBuffer CommandScheduler::CommandPool::Commit() {
	auto       gpu_tick = m_master.KnownGpuTick();
	const auto search   = [this, &gpu_tick](size_t begin, size_t end) -> std::optional<size_t> {
		for (size_t index = begin; index < end; ++index) {
			if (gpu_tick >= m_ticks[index]) {
				m_ticks[index] = m_master.CurrentTick();
				return index;
			}
		}
		return std::nullopt;
	};

	auto found = search(m_hint, m_ticks.size());
	if (!found) {
		m_master.Refresh();
		gpu_tick = m_master.KnownGpuTick();
		found    = search(m_hint, m_ticks.size());
	}
	if (!found) {
		found = search(0, m_hint);
	}
	if (!found) {
		found           = Grow();
		m_ticks[*found] = m_master.CurrentTick();
	}

	m_hint = (*found + 1) % m_ticks.size();
	return m_buffers[*found];
}

bool CommandScheduler::InDeferredOperation() noexcept {
	return g_deferred_callback_scheduler != nullptr;
}

CommandScheduler::CommandScheduler(RenderContext& context, GraphicContext& graphics)
    : m_master(graphics), m_context(context), m_graphics(graphics),
      m_command_pool(graphics, m_master), m_command(*this),
      m_priority_thread([this](std::stop_token stop) { PriorityOperationsThread(stop); }) {}

CommandScheduler::~CommandScheduler() {
	Shutdown();
}

void CommandScheduler::Shutdown() {
	{
		std::unique_lock lock(m_operation_mutex);
		if (m_operation_state == OperationState::Closed) {
			return;
		}
		if (g_deferred_callback_scheduler == this) {
			EXIT_IF(m_operation_state == OperationState::Open);
			// A priority callback cannot join its own runner, while a normal callback can be
			// executing inside the shutdown owner's final PopPendingOperations. The owning
			// thread will finish shutdown after this callback returns.
			return;
		}
		if (m_operation_state == OperationState::Draining) {
			m_operation_available.wait(
			    lock, [this] { return m_operation_state == OperationState::Closed; });
			return;
		}
		m_operation_state = OperationState::Draining;
	}
	if (!m_command.IsInvalid()) {
		Submit();
	}
	m_master.Wait(CurrentTick() - 1);
	PopPendingOperations();
	DrainPriorityOperations();
	m_priority_thread.request_stop();
	m_operation_available.notify_all();
	if (m_priority_thread.joinable()) {
		m_priority_thread.join();
	}
	{
		std::lock_guard lock(m_operation_mutex);
		EXIT_IF(!m_pending_operations.empty() || !m_priority_operations.empty() ||
		        m_priority_active_tick != 0);
		m_operation_state = OperationState::Closed;
	}
	m_operation_available.notify_all();
}

void CommandScheduler::Begin(HW::Context& registers, HW::UserConfig& user_config,
                             HW::Shader& shaders) {
	{
		std::lock_guard lock(m_operation_mutex);
		EXIT_IF(m_operation_state != OperationState::Open);
	}
	m_command.Bind(registers, user_config, shaders);

	if (m_command.IsInvalid()) {
		BeginNext();
	}
}

void CommandScheduler::Flush() {
	SubmitInfo submit;
	Flush(submit);
}

void CommandScheduler::Flush(SubmitInfo& submit) {
	Submit(submit);
	BeginNext();
}

void CommandScheduler::Finish() {
	CheckActive();
	if (!m_command.IsInvalid()) {
		Submit();
	}
	m_master.Wait(CurrentTick() - 1);
	BeginNext();
	PopPendingOperations();
}

void CommandScheduler::Wait(uint64_t tick) {
	struct Timed {
		std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
		~Timed() {
			BenchStats::AddSchedulerWait(static_cast<uint64_t>(
			    std::chrono::duration_cast<std::chrono::nanoseconds>(
			        std::chrono::steady_clock::now() - start)
			        .count()));
		}
	} timed;
	EXIT_IF(tick > CurrentTick());
	if (tick == CurrentTick()) {
		CheckActive();
		// A stream-buffer wrap can wait while a draw is being prepared through a reference to
		// Current(). The wrapper stays stable while its pooled Vulkan buffer is retired. Deferred
		// resources are released only at the next GPU operation boundary.
		const auto submitted_tick = Submit();
		EXIT_IF(submitted_tick != tick);
		m_master.Wait(tick);
		BeginNext();
	} else {
		m_master.Wait(tick);
	}
}

void CommandScheduler::PopPendingOperations() {
	m_master.Refresh();
	for (;;) {
		PendingOperation operation;
		{
			std::lock_guard lock(m_operation_mutex);
			if (m_pending_operations.empty() ||
			    !m_master.IsFree(m_pending_operations.front().tick)) {
				return;
			}
			operation = std::move(m_pending_operations.front());
			m_pending_operations.pop();
		}
		WaitPriorityOperations(operation.tick);
		RunOperation(std::move(operation.callback));
	}
}

void CommandScheduler::DeferOperation(Common::UniqueFunction<void>&& operation) {
	QueueOperation(std::move(operation), false);
}

void CommandScheduler::DeferPriorityOperation(Common::UniqueFunction<void>&& operation) {
	QueueOperation(std::move(operation), true);
}

void CommandScheduler::QueueOperation(Common::UniqueFunction<void>&& operation, bool priority) {
	CheckActive();
	EXIT_IF(!operation);
	std::unique_lock lock(m_operation_mutex);
	if (m_operation_state == OperationState::Open) {
		auto& queue = priority ? m_priority_operations : m_pending_operations;
		queue.push({std::move(operation), CurrentTick()});
		lock.unlock();
		if (priority) {
			m_operation_available.notify_one();
		}
		return;
	}
	if (g_deferred_callback_scheduler != this) {
		m_operation_available.wait(lock,
		                           [this] { return m_operation_state == OperationState::Closed; });
	}
	lock.unlock();
	operation();
}

bool CommandScheduler::HasPendingPriorityOperations() {
	std::lock_guard lock(m_operation_mutex);
	return !m_priority_operations.empty() || m_priority_active_tick != 0;
}

void CommandScheduler::PriorityOperationsThread(std::stop_token stop) {
	while (!stop.stop_requested()) {
		PendingOperation operation;
		{
			std::unique_lock lock(m_operation_mutex);
			m_operation_available.wait(lock, [this, &stop] {
				return stop.stop_requested() || !m_priority_operations.empty();
			});
			if (stop.stop_requested()) {
				return;
			}
			operation = std::move(m_priority_operations.front());
			m_priority_operations.pop();
			m_priority_active_tick = operation.tick;
		}
		m_master.Wait(operation.tick);
		if (!stop.stop_requested()) {
			RunOperation(std::move(operation.callback));
		}
		{
			std::lock_guard lock(m_operation_mutex);
			m_priority_active_tick = 0;
		}
		m_operation_available.notify_all();
	}
}

void CommandScheduler::DrainPriorityOperations() {
	EXIT_IF(g_deferred_callback_scheduler == this);
	std::unique_lock lock(m_operation_mutex);
	m_operation_available.wait(
	    lock, [this] { return m_priority_operations.empty() && m_priority_active_tick == 0; });
}

void CommandScheduler::WaitPriorityOperations(uint64_t tick) {
	EXIT_IF(g_deferred_callback_scheduler == this);
	std::unique_lock lock(m_operation_mutex);
	m_operation_available.wait(lock, [this, tick] {
		const bool active_before_or_at =
		    m_priority_active_tick != 0 && m_priority_active_tick <= tick;
		const bool queued_before_or_at =
		    !m_priority_operations.empty() && m_priority_operations.front().tick <= tick;
		return !active_before_or_at && !queued_before_or_at;
	});
}

void CommandScheduler::RunOperation(Common::UniqueFunction<void>&& operation) {
	auto* previous                = g_deferred_callback_scheduler;
	g_deferred_callback_scheduler = this;
	operation();
	g_deferred_callback_scheduler = previous;
}

bool CommandScheduler::IsFree(uint64_t tick) {
	if (m_master.IsFree(tick)) {
		return true;
	}
	m_master.Refresh();
	return m_master.IsFree(tick);
}

void CommandScheduler::CheckActive() const {
	EXIT_IF(!Active());
}

CommandBuffer& CommandScheduler::Current() {
	EXIT_IF(m_command.IsInvalid());
	return m_command;
}

CommandBuffer& CommandScheduler::BeginCommand() {
	EXIT_IF(!m_command.IsInvalid());
	m_command.m_buffer = m_command_pool.Commit();
	m_command.Begin();
	m_dynamic_state.Invalidate();
	if (BenchStats::Enabled()) {
		WriteStartTimestamp();
	}
	ProfileMark(ProfileKind::Start, 0, 0, 0, 0);
	return m_command;
}

void CommandScheduler::ProfileMark(ProfileKind kind, uint64_t hash, uint32_t a, uint32_t b,
                                   uint32_t wave) {
	static const int start = [] {
		const char* text = std::getenv("KYTY_GPU_PROFILE");
		return text != nullptr ? std::atoi(text) : -1;
	}();
	if (start < 0 || m_profile_done) {
		return;
	}
	static const auto began = std::chrono::steady_clock::now();
	if (m_profile_pool == nullptr) {
		if (kind != ProfileKind::Start ||
		    std::chrono::steady_clock::now() - began < std::chrono::seconds(start)) {
			return;
		}
		vk::QueryPoolCreateInfo info {};
		info.queryType  = vk::QueryType::eTimestamp;
		info.queryCount = ProfileMarks;
		EXIT_NOT_IMPLEMENTED(m_graphics.device.createQueryPool(&info, nullptr, &m_profile_pool) !=
		                     vk::Result::eSuccess);
		m_command.m_buffer.resetQueryPool(m_profile_pool, 0, ProfileMarks);
		m_profile.reserve(ProfileMarks);
	}
	m_command.m_buffer.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, m_profile_pool,
	                                  static_cast<uint32_t>(m_profile.size()));
	m_profile.push_back({hash, a, b, wave, kind});
	if (m_profile.size() == ProfileMarks) {
		m_profile_done = true;
		std::lock_guard lock(m_operation_mutex);
		m_pending_operations.push({[this] { ReportProfile(); }, CurrentTick()});
	}
}

void CommandScheduler::ReportProfile() {
	std::vector<uint64_t> stamps(m_profile.size());
	const auto result = m_graphics.device.getQueryPoolResults(
	    m_profile_pool, 0, static_cast<uint32_t>(stamps.size()), stamps.size() * sizeof(uint64_t),
	    stamps.data(), sizeof(uint64_t), vk::QueryResultFlagBits::e64);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
	const double period = m_graphics.GetPhysicalDeviceProperties().limits.timestampPeriod * 1e-6;
	struct Row {
		double   ms = 0.0;
		uint64_t count = 0, work = 0;
		uint32_t a = 0, b = 0, wave = 0;
		bool     draw = false;
	};
	std::map<uint64_t, Row> rows;
	double   draw_ms = 0.0, dispatch_ms = 0.0, other_ms = 0.0;
	uint64_t draws = 0, dispatches = 0;
	double   barrier_ms[ProfileBarriers] {};
	uint64_t barrier_count[ProfileBarriers] {};
	double   size_ms[6] {};
	uint64_t size_count[6] {};
	for (size_t i = 1; i < stamps.size(); i++) {
		const auto& record = m_profile[i];
		if (record.kind == ProfileKind::Start) continue;
		const double ms = static_cast<double>(stamps[i] - stamps[i - 1]) * period;
		if (record.kind == ProfileKind::Other) {
			other_ms += ms;
			continue;
		}
		if (record.kind == ProfileKind::Barrier) {
			barrier_ms[record.hash] += ms;
			barrier_count[record.hash]++;
			continue;
		}
		if (record.kind == ProfileKind::Dispatch) {
			const double us = ms * 1000.0;
			const int bucket = us < 5.0 ? 0 : us < 20.0 ? 1 : us < 100.0 ? 2 : us < 500.0 ? 3 : us < 2000.0 ? 4 : 5;
			size_ms[bucket] += ms;
			size_count[bucket]++;
		}
		auto&        row = rows[record.hash];
		row.ms += ms;
		row.count++;
		row.wave = record.wave;
		row.draw = record.kind == ProfileKind::Draw;
		if (row.draw) {
			row.a = record.a;
			row.b = record.b;
			draw_ms += ms;
			draws++;
		} else {
			row.work += uint64_t {record.a} * record.b;
			dispatch_ms += ms;
			dispatches++;
		}
	}
	const double frames = static_cast<double>(draws + dispatches) / 3950.0;
	std::printf("gpu-prof: frames=%.1f draws %.1f ms/frame, dispatches %.1f ms/frame\n",
	            frames, draw_ms / frames, dispatch_ms / frames);
	std::printf("gpu-prof: other %.1f ms/frame\n", other_ms / frames);
	const char* const barrier_names[ProfileBarriers] = {
	    "image-repeat-write", "image-same-layout", "image-layout-change",
	    "dispatch-conflict",  "dispatch-finish",   "run-start"};
	for (int i = 0; i < ProfileBarriers; i++) {
		std::printf("gpu-prof: barrier %-20s %.2f ms/frame %.0f/frame\n", barrier_names[i],
		            barrier_ms[i] / frames, static_cast<double>(barrier_count[i]) / frames);
	}
	const char* const size_names[6] = {"<5us", "<20us", "<100us", "<500us", "<2ms", ">2ms"};
	for (int i = 0; i < 6; i++) {
		std::printf("gpu-prof: dispatch %-7s %.2f ms/frame %.0f/frame\n", size_names[i],
		            size_ms[i] / frames, static_cast<double>(size_count[i]) / frames);
	}
	std::vector<std::pair<double, uint64_t>> order;
	for (const auto& [hash, row]: rows) order.emplace_back(row.ms, hash);
	std::sort(order.rbegin(), order.rend());
	for (size_t i = 0; i < std::min<size_t>(order.size(), 45u); i++) {
		const auto& row = rows[order[i].second];
		std::printf("gpu-prof: %s %016llx %.2f ms/frame %.1f ops/frame %.1f us/op wave=%u %s=%llu\n",
		            row.draw ? "draw" : "cs  ", static_cast<unsigned long long>(order[i].second),
		            row.ms / frames, static_cast<double>(row.count) / frames,
		            row.ms * 1000.0 / static_cast<double>(row.count), row.wave,
		            row.draw ? "target" : "groups/op",
		            row.draw ? static_cast<unsigned long long>(uint64_t {row.a} << 16u | row.b)
		                     : static_cast<unsigned long long>(row.work / row.count));
	}
	std::fflush(stdout);
}

void CommandScheduler::WriteStartTimestamp() {
	if (m_timestamp_pool == nullptr) {
		vk::QueryPoolCreateInfo info {};
		info.queryType  = vk::QueryType::eTimestamp;
		info.queryCount = TimestampSlots * 2;
		EXIT_NOT_IMPLEMENTED(m_graphics.device.createQueryPool(&info, nullptr, &m_timestamp_pool) !=
		                     vk::Result::eSuccess);
	}
	m_timestamp_slot = m_timestamp_next++ % TimestampSlots;
	m_command.m_buffer.resetQueryPool(m_timestamp_pool, m_timestamp_slot * 2, 2);
	m_command.m_buffer.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, m_timestamp_pool,
	                                  m_timestamp_slot * 2);
}

void CommandScheduler::WriteEndTimestamp() {
	const auto slot  = m_timestamp_slot;
	m_timestamp_slot = UINT32_MAX;
	m_command.m_buffer.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, m_timestamp_pool,
	                                  slot * 2 + 1);
	// Runs once the buffer's tick completes, in tick order.
	std::lock_guard lock(m_operation_mutex);
	m_pending_operations.push({[this, slot] { ReadTimestamps(slot); }, CurrentTick()});
}

void CommandScheduler::ReadTimestamps(uint32_t slot) {
	uint64_t   values[2] = {};
	const auto result    = m_graphics.device.getQueryPoolResults(
        m_timestamp_pool, slot * 2, 2, sizeof(values), values, sizeof(uint64_t),
        vk::QueryResultFlagBits::e64);
	if (result != vk::Result::eSuccess || values[1] < values[0]) {
		return;
	}
	const double period = m_graphics.GetPhysicalDeviceProperties().limits.timestampPeriod;
	const auto   to_ns  = [period](uint64_t ticks) {
        return static_cast<uint64_t>(static_cast<double>(ticks) * period);
	};
	const uint64_t gap = m_gpu_last_end != 0 && values[0] > m_gpu_last_end ? values[0] - m_gpu_last_end : 0u;
	const uint64_t from = std::max(values[0], m_gpu_last_end);
	BenchStats::AddGpuTime(values[1] > from ? to_ns(values[1] - from) : 0u, to_ns(gap));
	m_gpu_last_end = std::max(m_gpu_last_end, values[1]);
}

uint64_t CommandScheduler::Submit(SubmitInfo submit) {
	EXIT_IF(m_command.IsInvalid());

	// A timeline wait does not make GPU writes visible to the host, and reads of mapped game
	// buffers rely on one being enough. Every submission ends with that domain operation.
	m_command.EndRendering();
	vk::MemoryBarrier to_host {};
	to_host.srcAccessMask = vk::AccessFlagBits::eMemoryWrite;
	to_host.dstAccessMask = vk::AccessFlagBits::eHostRead;
	m_command.Handle().pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                                   vk::PipelineStageFlagBits::eHost, {}, 1, &to_host, 0,
	                                   nullptr, 0, nullptr);

	if (m_timestamp_slot != UINT32_MAX) {
		WriteEndTimestamp();
	}
	m_command.End();
	const auto buffer   = m_command.m_buffer;
	auto&      graphics = m_graphics;
	EXIT_IF(graphics.queue == nullptr);

	vk::Result result;
	uint64_t   tick;
	{
		Common::LockGuard lock(graphics.queue_mutex);
		tick = m_master.NextTick();
		submit.AddSignal(m_master.Handle(), tick);

		vk::TimelineSemaphoreSubmitInfo timeline_info {};
		timeline_info.waitSemaphoreValueCount   = submit.num_wait_semaphores;
		timeline_info.pWaitSemaphoreValues      = submit.wait_ticks.data();
		timeline_info.signalSemaphoreValueCount = submit.num_signal_semaphores;
		timeline_info.pSignalSemaphoreValues    = submit.signal_ticks.data();

		vk::SubmitInfo submit_info {};
		submit_info.pNext                = &timeline_info;
		submit_info.waitSemaphoreCount   = submit.num_wait_semaphores;
		submit_info.pWaitSemaphores      = submit.wait_semaphores.data();
		submit_info.pWaitDstStageMask    = submit.wait_stages.data();
		submit_info.commandBufferCount   = 1;
		submit_info.pCommandBuffers      = &buffer;
		submit_info.signalSemaphoreCount = submit.num_signal_semaphores;
		submit_info.pSignalSemaphores    = submit.signal_semaphores.data();

		result = graphics.queue.submit(1, &submit_info, nullptr);
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::QueueSubmits);

	if (result != vk::Result::eSuccess) {
		EXIT("vkQueueSubmit failed: %s, tick=%" PRIu64 "\n", vk::to_string(result).c_str(), tick);
	}

	m_command.m_buffer = nullptr;
	return tick;
}

void CommandScheduler::BeginNext() {
	CheckActive();
	BeginCommand();
}

} // namespace Libs::Graphics
