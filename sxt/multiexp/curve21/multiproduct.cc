#include "sxt/multiexp/curve21/multiproduct.h"

#include "sxt/algorithm/base/gather_mapper.h"
#include "sxt/base/macro/cuda_callable.h"
#include "sxt/curve21/operation/accumulator.h"
#include "sxt/curve21/operation/neg.h"
#include "sxt/execution/async/future.h"
#include "sxt/multiexp/multiproduct_gpu/multiproduct.h"

namespace sxt::mtxc21 {
//--------------------------------------------------------------------------------------------------
// signed_mapper
//--------------------------------------------------------------------------------------------------
namespace {
class signed_mapper {
public:
  using value_type = c21t::element_p3;

  CUDA_CALLABLE signed_mapper(const c21t::element_p3* generators, const unsigned* indexes) noexcept
      : generators_{generators}, indexes_{indexes} {}

  CUDA_CALLABLE void map_index(c21t::element_p3& val, unsigned int index) const noexcept {
    constexpr auto sign_bit = 1u << 31;
    auto i = indexes_[index];
    auto is_neg = static_cast<unsigned>((i & sign_bit) != 0);
    i = i & ~sign_bit;
    val = generators_[i];
    c21o::cneg(val, is_neg);
  }

  CUDA_CALLABLE c21t::element_p3 map_index(unsigned int index) const noexcept {
    c21t::element_p3 res;
    this->map_index(res, index);
    return res;
  }

private:
  const c21t::element_p3* generators_;
  const unsigned* indexes_;
};
} // namespace

//--------------------------------------------------------------------------------------------------
// async_compute_multiproduct
//--------------------------------------------------------------------------------------------------
xena::future<> async_compute_multiproduct(basct::span<c21t::element_p3> products,
                                          bast::raw_stream_t stream,
                                          basct::cspan<c21t::element_p3> generators,
                                          basct::cspan<unsigned> indexes,
                                          basct::cspan<unsigned> product_sizes,
                                          bool is_signed) noexcept {
  if (!is_signed) {
    using Mapper = algb::gather_mapper<c21t::element_p3>;
    return mtxmpg::compute_multiproduct<c21o::accumulator, Mapper>(products, stream, generators,
                                                                   indexes, product_sizes);
  }
  return mtxmpg::compute_multiproduct<c21o::accumulator, signed_mapper>(
      products, stream, generators, indexes, product_sizes);
}
} // namespace sxt::mtxc21
