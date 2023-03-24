#pragma once

#include <vector>

#include "sxt/execution/async/future.h"

namespace sxt::xena {
//--------------------------------------------------------------------------------------------------
// await_all
//--------------------------------------------------------------------------------------------------
// await multiple futures
//
// Note: This is to work around https://github.com/spaceandtimelabs/proofs-gpu/issues/200
future<> await_all(std::vector<future<>> futs) noexcept;

template <class... Tx>
  requires(std::is_same_v<Tx, future<>> && ...)
future<> await_all(Tx&&... futs) noexcept {
  std::vector<future<>> fut_vec;
  fut_vec.reserve(sizeof...(futs));
  (fut_vec.emplace_back(std::move(futs)), ...);
  return await_all(std::move(fut_vec));
}
} // namespace sxt::xena
