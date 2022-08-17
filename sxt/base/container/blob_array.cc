#include "sxt/base/container/blob_array.h"

namespace sxt::basct {
//--------------------------------------------------------------------------------------------------
// constructor
//--------------------------------------------------------------------------------------------------
blob_array::blob_array(size_t size, size_t blob_size) noexcept
    : blob_size_{blob_size == 0 ? 1 : blob_size}, data_(size * blob_size_) {}

//--------------------------------------------------------------------------------------------------
// resize
//--------------------------------------------------------------------------------------------------
void blob_array::resize(size_t size, size_t blob_size) noexcept {
  blob_size_ = blob_size == 0 ? 1 : blob_size;
  data_.resize(size * blob_size_);
}
} // namespace sxt::basct
