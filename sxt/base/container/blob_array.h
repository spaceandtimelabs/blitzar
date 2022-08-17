#pragma once

#include <cstdint>
#include <vector>

#include "sxt/base/container/span.h"
#include "sxt/base/container/span_iterator.h"

namespace sxt::basct {
//--------------------------------------------------------------------------------------------------
// blob_array
//--------------------------------------------------------------------------------------------------
class blob_array {
public:
  blob_array() noexcept = default;

  blob_array(size_t size, size_t blob_size) noexcept;

  uint8_t* data() noexcept { return data_.data(); }

  const uint8_t* data() const noexcept { return data_.data(); }

  bool empty() const noexcept { return data_.empty(); }

  size_t size() const noexcept { return data_.size() / blob_size_; }

  size_t blob_size() const noexcept { return blob_size_; }

  void resize(size_t size, size_t blob_size) noexcept;

  basct::span<uint8_t> operator[](size_t index) noexcept {
    return {
        data_.data() + index * blob_size_,
        blob_size_,
    };
  }

  basct::cspan<uint8_t> operator[](size_t index) const noexcept {
    return {
        data_.data() + index * blob_size_,
        blob_size_,
    };
  }

  span_iterator<uint8_t> begin() noexcept { return {data_.data(), blob_size_}; }

  span_iterator<uint8_t> end() noexcept { return {data_.data() + data_.size(), blob_size_}; }

  span_iterator<const uint8_t> begin() const noexcept { return {data_.data(), blob_size_}; }

  span_iterator<const uint8_t> end() const noexcept {
    return {data_.data() + data_.size(), blob_size_};
  }

private:
  size_t blob_size_{1};
  std::vector<uint8_t> data_;
};
} // namespace sxt::basct
