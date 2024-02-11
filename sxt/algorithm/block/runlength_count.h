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
    auto thread_id = threadIdx.x;
    for (unsigned i=thread_id; i<NumBins; i+=NumThreads) {
      storage_.run_begin[i] = NumThreads * ItemsPerThread;
      storage_.run_end[i] = NumThreads * ItemsPerThread;
    }
    int flags[ItemsPerThread];
    auto flag_op = [&storage = storage_] (T a, T b, int b_index) noexcept {
      storage.run_end[b] = static_cast<CounterT>(b_index); 
      storage.run_begin[b] = static_cast<CounterT>(b_index); 
    };
    __syncthreads();
    discontinuity_.FlagHeads(flags, items, flag_op);
    if (thread_id == 0) {
      storage_.run_begin[items[0]] = 0;
    }
    __syncthreads();
    for (unsigned i=thread_id; i<NumBins; i+=NumThreads) {
      counts[i] = storage_.run_end[i] - storage_.run_begin[i];
    }
  } 

private:
  temp_storage& storage_;
  Discontinuity discontinuity_;
};
} // namespace sxt::algbk
