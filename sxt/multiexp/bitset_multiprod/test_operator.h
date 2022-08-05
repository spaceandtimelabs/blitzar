#pragma once

#include <cstddef>
#include <cstdint>

namespace sxt::mtxbmp {
//--------------------------------------------------------------------------------------------------
// test_operator
//--------------------------------------------------------------------------------------------------
class test_operator {
public:
  test_operator(size_t* counter) noexcept : counter_{counter} {}

  void mark_unset(uint64_t& value) const noexcept { value = 0; }

  bool is_set(uint64_t value) const noexcept { return value != 0; }

  void add(uint64_t& res, uint64_t lhs, uint64_t rhs) const noexcept {
    ++*counter_;
    res = lhs + rhs;
  }

private:
  size_t* counter_;
};
} // namespace sxt::mtxbmp
