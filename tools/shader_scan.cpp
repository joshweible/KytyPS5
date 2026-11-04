// Finds PS5 GPU shader binaries in game files by their "barefoot" footer and hashes them the way
// ShaderMapUserData does, so a static shader list can be checked against hashes seen at runtime.
//
//   shader_scan <known_hashes.txt> <out.csv> <file-or-directory>...
//
// known_hashes.txt holds one hexadecimal shader hash per line. A binary's code does not always
// start exactly where the footer's sizes place it, so a few padded starts are tried and the one
// that matches a known hash wins.
#define XXH_INLINE_ALL
#include "xxhash.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;

constexpr size_t   FooterSize = 0x30;
constexpr uint32_t MaxPadding = 0x100;

static uint32_t Read32(const uint8_t* p) {
	uint32_t v;
	std::memcpy(&v, p, 4);
	return v;
}

static std::unordered_set<uint64_t> g_known;
static std::unordered_set<uint64_t> g_found_known;
static std::unordered_set<uint64_t> g_found_all;
static std::map<uint32_t, uint64_t> g_padding_histogram;
static uint64_t                     g_footers = 0, g_matches = 0, g_files_with_shaders = 0;

static void ScanBuffer(const std::vector<uint8_t>& data, const std::string& name, FILE* out) {
	bool any = false;
	for (size_t i = 0; i + FooterSize <= data.size();) {
		const void* hit = std::memchr(data.data() + i, 'b', data.size() - FooterSize - i + 1);
		if (hit == nullptr) break;
		const size_t pos = static_cast<const uint8_t*>(hit) - data.data();
		i                = pos + 1;
		if (std::memcmp(data.data() + pos, "barefoot", 8) != 0) continue;
		const auto version = Read32(data.data() + pos + 0x10);
		const auto code    = Read32(data.data() + pos + 0x14);
		const auto literal = Read32(data.data() + pos + 0x18);
		const auto extra   = Read32(data.data() + pos + 0x1c); // Metadata between literals and footer.
		if (version != 1 || code == 0 || code % 4 != 0) continue;
		const uint64_t end = ((uint64_t {code} + 15u) & ~uint64_t {15}) + literal;
		const uint64_t tail = (uint64_t {extra} + 15u) & ~uint64_t {15};
		if (end % 4 != 0 || end + tail > pos) continue;
		++g_footers;
		any = true;
		uint64_t first_hash = 0;
		bool     matched    = false;
		const size_t base  = pos - end - tail;
		size_t   start      = base;
		uint32_t padding    = 0;
		for (uint32_t pad = 0; pad <= MaxPadding && base >= pad; pad += 4) {
			auto hash = XXH3_64bits(data.data() + base - pad, end);
			// Builds before the program-size change hashed the whole binary, footer included.
			if (g_known.count(hash) == 0u) {
				const auto whole = XXH3_64bits(data.data() + base - pad, pos + FooterSize - (base - pad));
				if (g_known.count(whole) != 0u) hash = whole;
			}
			if (pad == 0) first_hash = hash;
			if (g_known.count(hash) != 0u) {
				first_hash = hash;
				start      = base - pad;
				padding    = pad;
				matched    = true;
				break;
			}
		}
		g_found_all.insert(first_hash);
		if (matched) {
			++g_matches;
			g_found_known.insert(first_hash);
			++g_padding_histogram[padding];
		}
		std::fprintf(out, "%016llx,%d,%zu,%u,%u,%u,%s\n",
		             static_cast<unsigned long long>(first_hash), matched ? 1 : 0, start, code,
		             literal, padding, name.c_str());
	}
	g_files_with_shaders += any ? 1u : 0u;
}

static void ScanFile(const fs::path& path, FILE* out) {
	std::error_code error;
	const auto      size = fs::file_size(path, error);
	if (error || size < FooterSize) return;
	std::ifstream file(path, std::ios::binary);
	std::vector<uint8_t> data(static_cast<size_t>(size));
	file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(size));
	ScanBuffer(data, path.generic_string(), out);
}

int main(int argc, char** argv) {
	if (argc < 4) {
		std::fprintf(stderr, "usage: shader_scan <known_hashes.txt> <out.csv> <path>...\n");
		return 1;
	}
	std::ifstream known(argv[1]);
	for (std::string line; std::getline(known, line);) {
		if (!line.empty()) g_known.insert(std::strtoull(line.c_str(), nullptr, 16));
	}
	FILE* out = std::fopen(argv[2], "wb");
	if (out == nullptr) return 1;
	std::fprintf(out, "hash,known,offset,code,literal,padding,file\n");
	uint64_t files = 0, bytes = 0;
	for (int i = 3; i < argc; ++i) {
		const fs::path root = argv[i];
		if (fs::is_directory(root)) {
			for (const auto& entry: fs::recursive_directory_iterator(root)) {
				if (!entry.is_regular_file()) continue;
				ScanFile(entry.path(), out);
				++files;
				bytes += entry.file_size();
			}
		} else {
			ScanFile(root, out);
			++files;
			bytes += fs::file_size(root);
		}
	}
	std::fclose(out);
	std::printf("scanned %llu files, %.2f GB\n", static_cast<unsigned long long>(files),
	            static_cast<double>(bytes) / 1e9);
	std::printf("footers=%llu in %llu files, unique shaders=%zu\n",
	            static_cast<unsigned long long>(g_footers),
	            static_cast<unsigned long long>(g_files_with_shaders), g_found_all.size());
	std::printf("known hashes=%zu, found in files=%zu (%.1f%%)\n", g_known.size(),
	            g_found_known.size(),
	            g_known.empty() ? 0.0 : 100.0 * g_found_known.size() / g_known.size());
	for (const auto& [padding, count]: g_padding_histogram) {
		std::printf("  padding %u: %llu\n", padding, static_cast<unsigned long long>(count));
	}
	return 0;
}
