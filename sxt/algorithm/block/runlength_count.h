#pragma once

#include "cub/cub.cuh"

namespace sxt::algbk {
//--------------------------------------------------------------------------------------------------
// runlength_count 
//--------------------------------------------------------------------------------------------------
template <class T, class CounterT, unsigned NumThreads, unsigned NumBins>
class runlength_count {
  using Discontinuity = cub::BlockDiscontinuity<T, NumThreads>;
public:
  struct temp_storage {
    typename Discontinuity::TempStorage discontinuity;
    CounterT run_begin[NumBins];
    CounterT run_end[NumBins];
  };

  explicit runlength_count(temp_storage& storage) noexcept
      : storage_{storage}, discontinuity_{storage.discontinuity} {}

  template <unsigned ItemsPerThread>
  void count(CounterT counts[NumBins], T (&items)[ItemsPerThread]) noexcept {
  } 

private:
  temp_storage& storage_;
  Discontinuity discontinuity_;
};
} // namespace sxt::algbk
