/**
 * Adopt from Daniel Lemire's blog post
 *
 * https://lemire.me/blog/2018/03/08/iterating-over-set-bits-quickly-simd-edition/
 *
 * and
 *
 * https://github.com/lemire/Code-used-on-Daniel-Lemire-s-blog/tree/master/2018/03/07
 */
#include "sxt/base/bit/bit_position.h"

#include <x86intrin.h>

#include <iterator>

#include "sxt/base/bit/bit_position_decode_table.h"
#include "sxt/base/bit/span_op.h"
#include "sxt/base/error/assert.h"

namespace sxt::basbt {
//--------------------------------------------------------------------------------------------------
// compute_bit_positions_avx2_impl
//--------------------------------------------------------------------------------------------------
static void compute_bit_positions_avx2_impl(uint32_t*& out, const uint64_t* array,
                                            size_t sizeinwords, int offset) noexcept {
  __m256i base_vec = _mm256_set1_epi32(offset - 1);
  __m256i inc_vec = _mm256_set1_epi32(64);
  __m256i add8 = _mm256_set1_epi32(8);

  for (size_t i = 0; i < sizeinwords; ++i) {
    uint64_t w = array[i];
    if (w == 0) {
      base_vec = _mm256_add_epi32(base_vec, inc_vec);
    } else {
      for (int k = 0; k < 4; ++k) {
        uint8_t byte_a = (uint8_t)w;
        uint8_t byte_b = (uint8_t)(w >> 8);
        w >>= 16;
        __m256i vec_a = _mm256_cvtepu8_epi32(
            _mm_cvtsi64_si128(*(uint64_t*)(bit_position_decode_table[byte_a])));
        __m256i vec_b = _mm256_cvtepu8_epi32(
            _mm_cvtsi64_si128(*(uint64_t*)(bit_position_decode_table[byte_b])));
        uint8_t advance_a = __builtin_popcount(byte_a);
        uint8_t advance_b = __builtin_popcount(byte_b);
        vec_a = _mm256_add_epi32(base_vec, vec_a);
        base_vec = _mm256_add_epi32(base_vec, add8);
        vec_b = _mm256_add_epi32(base_vec, vec_b);
        base_vec = _mm256_add_epi32(base_vec, add8);
        _mm256_storeu_si256((__m256i*)out, vec_a);
        out += advance_a;
        _mm256_storeu_si256((__m256i*)out, vec_b);
        out += advance_b;
      }
    }
  }
}

//--------------------------------------------------------------------------------------------------
// compute_bit_positions_portable
//--------------------------------------------------------------------------------------------------
static void compute_bit_positions_portable(basct::span<unsigned>& positions,
                                           basct::cspan<uint8_t> blob, unsigned offset) noexcept {
  size_t cnt = 0;
  for_each_bit(blob, [&](unsigned pos) noexcept { positions[cnt++] = pos + offset; });
  positions = {
      positions.data(),
      cnt,
  };
}

//--------------------------------------------------------------------------------------------------
// compute_bit_positions_avx2
//--------------------------------------------------------------------------------------------------
void compute_bit_positions_avx2(basct::span<unsigned>& positions,
                                basct::cspan<uint8_t> blob) noexcept {
  if (blob.size() < 16) {
    return compute_bit_positions_portable(positions, blob, 0);
  }
  auto first = blob.data();
  auto offset = reinterpret_cast<uint64_t>(first) % 8;
  auto first_p = first;
  if (offset > 0) {
    first_p += 8 - offset;
  }
  auto last = first + blob.size();
  offset = reinterpret_cast<uint64_t>(last) % 8;
  auto last_p = last;
  if (offset > 0) {
    last_p -= offset;
  }
  SXT_DEBUG_ASSERT(std::distance(first_p, last_p) % 8 == 0);
  auto positions_p = positions;
  compute_bit_positions_portable(positions_p, blob.subspan(0, std::distance(first, first_p)), 0);
  auto out = positions_p.end();
  compute_bit_positions_avx2_impl(out, reinterpret_cast<const uint64_t*>(first_p),
                                  std::distance(first_p, last_p) / 8,
                                  static_cast<int>(std::distance(first, first_p) * 8));
  positions_p = {out, static_cast<size_t>(std::distance(out, positions.end()))};
  compute_bit_positions_portable(positions_p, blob.subspan(std::distance(first, last_p)),
                                 static_cast<unsigned>(std::distance(first, last_p) * 8));
  positions = {
      positions.data(),
      static_cast<size_t>(std::distance(positions.begin(), positions_p.end())),
  };
}

//--------------------------------------------------------------------------------------------------
// compute_bit_positions
//--------------------------------------------------------------------------------------------------
void compute_bit_positions(basct::span<unsigned>& positions, basct::cspan<uint8_t> blob) noexcept {
  compute_bit_positions_avx2(positions, blob);
}
} // namespace sxt::basbt
