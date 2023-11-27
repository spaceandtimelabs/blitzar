#pragma once

#include <memory_resource>
#include <vector>
#include <tuple>

namespace sxt::memr {
//--------------------------------------------------------------------------------------------------
// chained_resource
//--------------------------------------------------------------------------------------------------
class chained_resource final : public std::pmr::memory_resource {
public:
  chained_resource() noexcept;

  explicit chained_resource(std::pmr::memory_resource* upstream) noexcept;

  ~chained_resource() noexcept;

private:
  std::pmr::memory_resource* upstream_;
  std::vector<std::tuple<void*, size_t, size_t>> allocations_;

  void* do_allocate(size_t bytes, size_t alignment) noexcept override;

  void do_deallocate(void* /*ptr*/, size_t /*bytes*/, size_t /*alignment*/) noexcept {
    // destructor does bulk deallocation
  }

  bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override;
};
} // namespace sxt::memr
