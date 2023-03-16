#include "sxt/execution/async/test_kernel.h"

#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/property.h"
#include "sxt/base/num/divide_up.h"
#include "sxt/execution/base/stream.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"

namespace sxt::xena {
//--------------------------------------------------------------------------------------------------
// add_impl
//--------------------------------------------------------------------------------------------------
static __global__ void add_impl(uint64_t* c, const uint64_t* a, const uint64_t* b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

//--------------------------------------------------------------------------------------------------
// add_for_testing
//--------------------------------------------------------------------------------------------------
void add_for_testing(uint64_t* c, bast::raw_stream_t stream, const uint64_t* a, const uint64_t* b,
                     int n) noexcept {
  memr::async_device_resource resource{stream};

  memmg::managed_array<uint64_t> a_dev{&resource};
  memmg::managed_array<uint64_t> b_dev{&resource};
  memmg::managed_array<uint64_t> c_dev{&resource};
  auto cp = c;
  if (!basdv::is_device_pointer(a)) {
    a_dev = memmg::managed_array<uint64_t>{static_cast<unsigned>(n), &resource};
    basdv::async_memcpy_host_to_device(a_dev.data(), a, n * sizeof(uint64_t), stream);
    a = a_dev.data();
  }
  if (!basdv::is_device_pointer(b)) {
    b_dev = memmg::managed_array<uint64_t>{static_cast<unsigned>(n), &resource};
    basdv::async_memcpy_host_to_device(b_dev.data(), b, n * sizeof(uint64_t), stream);
    b = b_dev.data();
  }
  if (!basdv::is_device_pointer(c)) {
    c_dev = memmg::managed_array<uint64_t>{static_cast<unsigned>(n), &resource};
    cp = c_dev.data();
  }
  add_impl<<<basn::divide_up(n, 256), 256, 0, stream>>>(cp, a, b, n);
  if (!basdv::is_device_pointer(c)) {
    basdv::async_memcpy_device_to_host(c, cp, n * sizeof(uint64_t), stream);
  }
}
} // namespace sxt::xena
