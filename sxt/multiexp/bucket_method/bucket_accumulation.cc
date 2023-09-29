#include "sxt/multiexp/bucket_method/bucket_accumulation.h"

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// make_exponents_viewable
//--------------------------------------------------------------------------------------------------
basct::cspan<const uint8_t>
make_exponents_viewable(memmg::managed_array<uint8_t>& exponents_viewable_data,
                        basct::cspan<const uint8_t*> exponents, const basit::index_range& rng,
                        const basdv::stream& stream) noexcept {
  static constexpr size_t exponent_size = 32; // hard coded for now
  auto num_outputs = exponents.size();
  auto n = rng.size();
  exponents_viewable_data = memmg::managed_array<uint8_t>{exponent_size * n * num_outputs,
                                                          exponents_viewable_data.get_allocator()};
  auto out = exponents_viewable_data.data();
  for (size_t output_index=0; output_index<num_outputs; ++output_index) {
    basdv::async_copy_to_device(
        basct::span<uint8_t>{out, n * exponent_size},
        basct::cspan<uint8_t>{exponents[output_index] + rng.a() * exponent_size, n * exponent_size},
        stream);
    out += n * exponent_size;
  }
  return exponents_viewable_data;
}
} // namespace sxt::mtxbk
