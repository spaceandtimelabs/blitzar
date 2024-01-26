#include <print>

#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/device/synchronization.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/device_resource.h"
using namespace sxt;

/* using T = unsigned; */
using T = unsigned short;

__global__ void f(T* out, T n) {
  auto thread_index = threadIdx.x;
  auto num_threads = gridDim.x;
  extern __shared__ T cnts[];
  for (unsigned i=thread_index; i<n; i += num_threads) {
    cnts[i] = i;
  }
  for (unsigned i=thread_index; i<n; i += num_threads) {
    out[i] = cnts[i];
  }
}

int main() {
  T n = 20'000;
  basdv::stream stream;
  memmg::managed_array<T> v_dev{n, memr::get_device_resource()};
  f<<<1, 32, sizeof(T) * n, stream>>>(v_dev.data(), n);
  memmg::managed_array<T> v(n);
  basdv::async_copy_device_to_host(v, v_dev, stream);
  basdv::synchronize_stream(stream);
  std::print("{}\n", v[n-1]);
  return 0;
}
