// Summarizes a --shader-cache store: one CSV row per module record, to measure how much of a
// shader's final SPIR-V depends on its code and static key rather than runtime specialization.
//
//   store_scan <PPSA26344.shaders> <out.csv> <stage_enum_bytes>
#define XXH_INLINE_ALL
#include "xxhash.h"
#include "zstd.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

struct Frame {
	uint32_t compressed_size;
	uint32_t raw_size;
	uint64_t checksum;
};

static bool FrameAt(const std::vector<uint8_t>& b, size_t at, Frame& f) {
	if (at + sizeof(Frame) > b.size()) return false;
	std::memcpy(&f, b.data() + at, sizeof(f));
	if (f.compressed_size == 0 || f.raw_size == 0 || at + sizeof(f) + f.compressed_size > b.size())
		return false;
	return XXH3_64bits(b.data() + at + sizeof(f), f.compressed_size) == f.checksum;
}

int main(int argc, char** argv) {
	if (argc < 4) return 1;
	const size_t  stage_bytes = static_cast<size_t>(std::atoi(argv[3]));
	std::ifstream in(argv[1], std::ios::binary | std::ios::ate);
	std::vector<uint8_t> b(static_cast<size_t>(in.tellg()));
	in.seekg(0);
	in.read(reinterpret_cast<char*>(b.data()), static_cast<std::streamsize>(b.size()));
	size_t at = 0;
	Frame  f {};
	while (at < 4096 && !FrameAt(b, at, f)) ++at;
	if (at == 4096) {
		std::printf("no frame found\n");
		return 1;
	}
	std::printf("first frame at %zu of %zu bytes\n", at, b.size());
	FILE* out = std::fopen(argv[2], "wb");
	std::fprintf(out, "stage,hash,user_data,static_len,static_hash,plan_bytes,plan_hash,meta_bytes,"
	                  "meta_hash,spirv_bytes,spirv_hash,static_hex\n");
	uint64_t modules = 0, pipelines = 0, layouts = 0, other = 0;
	std::vector<uint8_t> raw;
	while (FrameAt(b, at, f)) {
		raw.resize(f.raw_size);
		ZSTD_decompress(raw.data(), raw.size(), b.data() + at + sizeof(f), f.compressed_size);
		at += sizeof(f) + f.compressed_size;
		const uint8_t type = raw[0];
		if (type == 1) {
			++pipelines;
			continue;
		}
		if (type == 2) {
			++layouts;
			continue;
		}
		if (type != 0) {
			++other;
			continue;
		}
		++modules;
		size_t   p     = 1;
		uint64_t stage = 0;
		std::memcpy(&stage, raw.data() + p, stage_bytes);
		p += stage_bytes;
		uint64_t hash;
		std::memcpy(&hash, raw.data() + p, 8);
		p += 8;
		uint32_t user_data, static_len;
		std::memcpy(&user_data, raw.data() + p, 4);
		p += 4;
		std::memcpy(&static_len, raw.data() + p, 4);
		p += 4;
		const auto* static_words = raw.data() + p;
		const auto  static_hash  = XXH3_64bits(static_words, static_len * 4u);
		p += static_len * 4u;
		uint32_t plan_bytes;
		std::memcpy(&plan_bytes, raw.data() + p, 4);
		p += 4;
		const auto plan_hash = XXH3_64bits(raw.data() + p, plan_bytes);
		p += plan_bytes;
		// Module id, then metadata, then the SPIR-V vector ends the record.
		p += 8;
		size_t spirv = 0;
		for (size_t i = raw.size() - 4; i > p; --i) {
			uint32_t word;
			std::memcpy(&word, raw.data() + i, 4);
			if (word != 0x07230203u) continue;
			uint32_t count;
			std::memcpy(&count, raw.data() + i - 4, 4);
			if (count * 4u == raw.size() - i) {
				spirv = i;
				break;
			}
		}
		const size_t meta      = spirv >= p + 4 ? spirv - 4 - p : 0;
		const auto   meta_hash = XXH3_64bits(raw.data() + p, meta);
		const auto   spv_hash  = XXH3_64bits(raw.data() + spirv, raw.size() - spirv);
		if (argc > 4 && spirv != 0 && stage == 3u) {
			char name[512];
			std::snprintf(name, sizeof(name), "%s/%016llx_%zu.spv", argv[4],
			              static_cast<unsigned long long>(hash), raw.size() - spirv);
			if (FILE* dump = std::fopen(name, "wb")) {
				std::fwrite(raw.data() + spirv, 1, raw.size() - spirv, dump);
				std::fclose(dump);
			}
		}
		std::fprintf(out, "%llu,%016llx,%u,%u,%016llx,%u,%016llx,%zu,%016llx,%zu,%016llx,",
		             static_cast<unsigned long long>(stage), static_cast<unsigned long long>(hash),
		             user_data, static_len, static_cast<unsigned long long>(static_hash),
		             plan_bytes, static_cast<unsigned long long>(plan_hash), meta,
		             static_cast<unsigned long long>(meta_hash), raw.size() - spirv,
		             static_cast<unsigned long long>(spv_hash));
		for (uint32_t i = 0; i < static_len && i < 64u; ++i) {
			uint32_t word;
			std::memcpy(&word, static_words + i * 4u, 4);
			std::fprintf(out, "%08x", word);
		}
		std::fprintf(out, "\n");
	}
	std::fclose(out);
	std::printf("modules=%llu pipelines=%llu layouts=%llu other=%llu stopped at %zu\n",
	            static_cast<unsigned long long>(modules),
	            static_cast<unsigned long long>(pipelines),
	            static_cast<unsigned long long>(layouts), static_cast<unsigned long long>(other), at);
	return 0;
}
