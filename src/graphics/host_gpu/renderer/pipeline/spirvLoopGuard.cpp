#include "graphics/host_gpu/renderer/pipeline/spirvLoopGuard.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>

namespace Libs::Graphics {

namespace {

constexpr uint32_t OpEntryPoint        = 15;
constexpr uint32_t OpTypeBool          = 20;
constexpr uint32_t OpTypeInt           = 21;
constexpr uint32_t OpTypePointer       = 32;
constexpr uint32_t OpConstant          = 43;
constexpr uint32_t OpFunction          = 54;
constexpr uint32_t OpFunctionEnd       = 56;
constexpr uint32_t OpVariable          = 59;
constexpr uint32_t OpLoad              = 61;
constexpr uint32_t OpStore             = 62;
constexpr uint32_t OpIAdd              = 128;
constexpr uint32_t OpUGreaterThan      = 172;
constexpr uint32_t OpPhi               = 245;
constexpr uint32_t OpLoopMerge         = 246;
constexpr uint32_t OpSelectionMerge    = 247;
constexpr uint32_t OpLabel             = 248;
constexpr uint32_t OpBranch            = 249;
constexpr uint32_t OpBranchConditional = 250;
constexpr uint32_t OpReturn            = 253;
constexpr uint32_t StorageFunction     = 7;

void Emit(std::vector<uint32_t>& out, uint32_t op, std::initializer_list<uint32_t> operands) {
	out.push_back(static_cast<uint32_t>(operands.size() + 1u) << 16u | op);
	out.insert(out.end(), operands);
}

struct Options {
	uint64_t hash  = 0;
	uint32_t first = 0;
	uint32_t last  = UINT32_MAX;
	uint32_t limit = 1u << 16u;
};

// KYTY_LOOP_GUARD=<hash>[:first:last[:limit]]
const Options& GuardOptions() {
	static const Options options = [] {
		Options result;
		const char* text = std::getenv("KYTY_LOOP_GUARD");
		if (text == nullptr) return result;
		char* end   = nullptr;
		result.hash = std::strtoull(text, &end, 16);
		if (*end == ':') result.first = static_cast<uint32_t>(std::strtoul(end + 1, &end, 10));
		if (*end == ':') result.last = static_cast<uint32_t>(std::strtoul(end + 1, &end, 10));
		if (*end == ':') result.limit = static_cast<uint32_t>(std::strtoul(end + 1, &end, 10));
		return result;
	}();
	return options;
}

} // namespace

std::vector<uint32_t> GuardLoops(const std::vector<uint32_t>& spirv, uint32_t first, uint32_t last,
                                 uint32_t limit) {
	constexpr size_t HeaderWords = 5;
	uint32_t         bound       = spirv[3];
	uint32_t         entry       = 0;
	uint32_t         type_uint   = 0;
	uint32_t         type_bool   = 0;
	uint32_t         type_ptr    = 0;
	size_t           first_function = 0;

	for (size_t i = HeaderWords; i < spirv.size();) {
		const auto op    = spirv[i] & 0xffffu;
		const auto words = spirv[i] >> 16u;
		if (op == OpEntryPoint) {
			entry = spirv[i + 2];
		} else if (op == OpTypeBool) {
			type_bool = spirv[i + 1];
		} else if (op == OpTypeInt && spirv[i + 2] == 32u && spirv[i + 3] == 0u) {
			type_uint = spirv[i + 1];
		} else if (op == OpFunction) {
			first_function = i;
			break;
		}
		i += words;
	}
	for (size_t i = HeaderWords; i < first_function;) {
		if ((spirv[i] & 0xffffu) == OpTypePointer && spirv[i + 2] == StorageFunction &&
		    spirv[i + 3] == type_uint) {
			type_ptr = spirv[i + 1];
		}
		i += spirv[i] >> 16u;
	}
	if (entry == 0 || type_uint == 0 || type_bool == 0 || first_function == 0) return spirv;

	const auto constant_zero  = bound++;
	const auto constant_one   = bound++;
	const auto constant_limit = bound++;
	std::vector<uint32_t> out(spirv.begin(), spirv.begin() + static_cast<ptrdiff_t>(first_function));
	if (type_ptr == 0) {
		type_ptr = bound++;
		Emit(out, OpTypePointer, {type_ptr, StorageFunction, type_uint});
	}
	Emit(out, OpConstant, {type_uint, constant_zero, 0u});
	Emit(out, OpConstant, {type_uint, constant_one, 1u});
	Emit(out, OpConstant, {type_uint, constant_limit, limit});

	// First pass over the entry function: which loops are guarded, and their counters.
	struct Guard {
		uint32_t counter = 0;
		uint32_t resume  = 0; // Block that takes over the header's terminator.
	};
	std::unordered_map<size_t, Guard>    guards;    // By OpLoopMerge offset.
	std::unordered_map<uint32_t, uint32_t> renamed; // Header label to resume label, for Phi edges.
	std::vector<uint32_t>                counters;
	size_t   entry_begin = 0;
	size_t   entry_end   = 0;
	uint32_t loop_index  = 0;
	uint32_t label       = 0;
	for (size_t i = first_function; i < spirv.size();) {
		const auto op = spirv[i] & 0xffffu;
		if (op == OpFunction && spirv[i + 2] == entry) entry_begin = i;
		if (op == OpFunctionEnd && entry_begin != 0 && entry_end == 0) entry_end = i;
		if (entry_begin != 0 && entry_end == 0) {
			if (op == OpLabel) label = spirv[i + 1];
			if (op == OpLoopMerge) {
				const auto index = loop_index++;
				if (index >= first && index <= last) {
					Guard guard {.counter = bound++, .resume = bound++};
					counters.push_back(guard.counter);
					guards.emplace(i, guard);
					renamed.emplace(label, guard.resume);
				}
			}
		}
		i += spirv[i] >> 16u;
	}

	bool     entry_label_seen = false;
	bool     variables_placed = false;
	for (size_t i = first_function; i < spirv.size();) {
		const auto op    = spirv[i] & 0xffffu;
		const auto words = spirv[i] >> 16u;
		const bool in_entry = i >= entry_begin && i < entry_end;
		if (in_entry && entry_label_seen && !variables_placed && op != OpVariable) {
			for (const auto counter: counters) {
				Emit(out, OpVariable, {type_ptr, counter, StorageFunction, constant_zero});
			}
			variables_placed = true;
		}
		if (in_entry && op == OpLabel) entry_label_seen = true;

		if (in_entry && op == OpPhi) {
			const auto start = out.size();
			out.insert(out.end(), spirv.begin() + static_cast<ptrdiff_t>(i),
			           spirv.begin() + static_cast<ptrdiff_t>(i + words));
			for (uint32_t k = 4; k + 1u <= words; k += 2u) {
				const auto parent = out[start + k];
				if (const auto found = renamed.find(parent); found != renamed.end()) {
					out[start + k] = found->second;
				}
			}
			// A back edge into the header keeps its parent; only the header's own edges move, and
			// the header ends in OpBranch to the new block, so its successors see the resume block.
			i += words;
			continue;
		}

		out.insert(out.end(), spirv.begin() + static_cast<ptrdiff_t>(i),
		           spirv.begin() + static_cast<ptrdiff_t>(i + words));
		if (const auto found = guards.find(i); found != guards.end()) {
			const auto& guard   = found->second;
			const auto  check   = bound++;
			const auto  bail    = bound++;
			const auto  loaded  = bound++;
			const auto  counted = bound++;
			const auto  over    = bound++;
			Emit(out, OpBranch, {check});
			Emit(out, OpLabel, {check});
			Emit(out, OpLoad, {type_uint, loaded, guard.counter});
			Emit(out, OpIAdd, {type_uint, counted, loaded, constant_one});
			Emit(out, OpStore, {guard.counter, counted});
			Emit(out, OpUGreaterThan, {type_bool, over, counted, constant_limit});
			Emit(out, OpSelectionMerge, {guard.resume, 0u});
			Emit(out, OpBranchConditional, {over, bail, guard.resume});
			Emit(out, OpLabel, {bail});
			Emit(out, OpReturn, {});
			Emit(out, OpLabel, {guard.resume});
			// The header's terminator follows in the input and lands in the resume block.
		}
		i += words;
	}
	out[3] = bound;
	std::printf("loop-guard: guarded %zu of %u loops at limit %u\n", counters.size(), loop_index,
	            limit);
	std::fflush(stdout);
	return out;
}

std::vector<uint32_t> ApplyLoopGuard(uint64_t shader_hash, const std::vector<uint32_t>& spirv) {
	const auto& options = GuardOptions();
	if (options.hash == 0 || options.hash != shader_hash) return spirv;
	return GuardLoops(spirv, options.first, options.last, options.limit);
}

} // namespace Libs::Graphics
