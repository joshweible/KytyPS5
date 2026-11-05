#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRUN_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRUN_H_

#include "graphics/guest_gpu/command_processor/pm4Dispatch.h"

#include "common/abi.h"
#include "common/common.h"
#include "common/threads.h"
#include "common/uniqueFunction.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <semaphore>
#include <span>

namespace Libs::Graphics {

class RenderContext;

class GuestGpu final {
public:
	explicit GuestGpu(RenderContext& renderer);
	~GuestGpu();
	KYTY_CLASS_NO_COPY(GuestGpu);

	void               Shutdown();
	[[nodiscard]] bool IsStopping();
	void               SendCommand(Common::UniqueFunction<void>&& command);
	void               SendCommandSync(Common::UniqueFunction<void>&& command);

	// Submitted command memory is borrowed and must remain valid until GPU execution completes.
	void Submit(std::span<const uint32_t> commands);
	void SubmitCompute(uint32_t queue, std::span<const uint32_t> commands);
	void SubmitFlipPreparation(uint64_t request_id);
	// Insert an ordered graphics drain; only a previous suspend point can block the caller.
	void SuspendPoint();
	// Wait for guest command processing, including all compute queues (not native GPU completion).
	void              WaitForIdle();
	[[nodiscard]] int GetFrameNum() const;

	[[nodiscard]] static bool IsGpuThread() noexcept;

private:
	static constexpr uint32_t ComputePipeCount     = 7;
	static constexpr uint32_t QueuesPerComputePipe = 8;
	static constexpr uint32_t ComputeQueueCount    = ComputePipeCount * QueuesPerComputePipe;
	static constexpr uint32_t ComputeQueueBase     = 0x20;
	static constexpr uint32_t QueueCount           = 1 + ComputeQueueCount;

	enum class SubmissionType { Commands, FlipPreparation, SuspendPoint };

	struct Submission {
		SubmissionType            type     = SubmissionType::Commands;
		uint32_t                  queue_id = 0;
		std::span<const uint32_t> commands;
		Pm4Execution              execution;
		bool                      blocked         = false;
		uint64_t                  flip_request_id = 0;
	};

	struct Op;

	void              Enqueue(Submission submission);
	void              ProcessCommands();
	bool              Process(Submission& submission);
	static void       ThreadRun(void* data);
	static void       FrontRun(void* data);
	CommandProcessor& GetProcessor(uint32_t queue_id);

	// The GPU thread runs these in order. Only the front thread adds them.
	Op&      NextOp();
	void     PublishOp();
	bool     RunNextOp();
	void     DrainOps();
	void     WaitForOp(uint32_t target);
	void     WakeBack();
	void     SyncWrites() {
		if (m_unsynced_writes) {
			DrainOps();
		}
	}
	void     PostCall(CommandProcessor* state, Common::UniqueFunction<void>&& call);
	void     PostPacket(CommandProcessor& back, CommandProcessor* state,
	                    cp_op_parser_func_t handler, uint32_t cmd, const uint32_t* buffer,
	                    uint32_t dw, uint32_t num_dw, bool writes_memory);
	uint32_t RunPacket(CommandProcessor& cp, cp_op_parser_func_t handler, uint32_t cmd,
	                   const uint32_t* buffer, uint32_t dw, uint32_t num_dw);

	RenderContext&                                 m_renderer;
	Common::Mutex                                  m_queue_mutex;
	std::mutex                                     m_shutdown_mutex;
	Common::CondVar                                m_work_available;
	Common::CondVar                                m_idle;
	std::array<std::deque<Submission>, QueueCount> m_queues;
	std::deque<Common::UniqueFunction<void>>       m_commands;
	std::atomic_uint32_t                           m_pending_commands {0};
	uint32_t                                       m_next_queue        = 0;
	uint32_t                                       m_submission_count  = 0;
	bool                                           m_processing        = false;
	bool                                           m_command_running   = false;
	bool                                           m_accepting         = true;
	bool                                           m_stopping          = false;
	bool                                           m_shutdown_complete = false;
	// Completion callbacks can outlive GuestGpu during renderer shutdown.
	std::shared_ptr<std::binary_semaphore> m_suspend_point_ready =
	    std::make_shared<std::binary_semaphore>(1);

	std::unique_ptr<CommandProcessor>                                m_gfx_cp;
	std::array<std::unique_ptr<CommandProcessor>, ComputeQueueCount> m_compute_cp;
	std::unique_ptr<CommandProcessor>                                m_gfx_back;
	std::array<std::unique_ptr<CommandProcessor>, ComputeQueueCount> m_compute_back;

	std::unique_ptr<Op[]> m_ops;
	// Each thread works from its own copy of the other's counter until that runs out, so the two
	// don't trade these lines on every op.
	alignas(64) std::atomic_uint32_t m_op_write {0};
	uint32_t m_front_read = 0;
	alignas(64) std::atomic_uint32_t m_op_read {0}; // Advances when an op has finished.
	uint32_t m_back_write = 0;
	alignas(64) std::atomic_uint32_t m_back_wake {0};
	alignas(64) std::atomic_bool m_back_sleeping {false};
	alignas(64) std::atomic_bool m_front_sleeping {false};
	std::atomic_uint32_t m_front_target {0}; // The op the front thread sleeps until.
	alignas(64) char m_ops_end = 0; // Keeps what follows off the line above.
	std::atomic_bool      m_back_stopping {false};
	bool                  m_unsynced_writes = false; // Queued packets write guest memory.

	std::atomic_int m_done_num = 0;
	std::unique_ptr<Common::ThreadWithStack> m_thread;
	std::unique_ptr<Common::ThreadWithStack> m_front_thread;

	friend class CommandProcessor;
};
} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRUN_H_ */
