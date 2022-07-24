#pragma once

#include <cstddef>

namespace sxt::c21t {
struct element_p3;
}

namespace sxt::mtxrs {
//--------------------------------------------------------------------------------------------------
// input_accessor
//--------------------------------------------------------------------------------------------------
class input_accessor {
public:
  virtual ~input_accessor() noexcept = default;

  virtual void get_element(c21t::element_p3& p, const void* data, size_t index) const noexcept = 0;
};
} // namespace sxt::mtxrs
