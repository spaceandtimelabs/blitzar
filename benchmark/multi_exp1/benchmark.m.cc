#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string_view>

#include "benchmark/multi_exp1/multi_exp_cpu.h"
#include "benchmark/multi_exp1/multi_exp_gpu.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/memory/management/managed_array.h"

using namespace sxt;

using bench_fn = void (*)(c21t::element_p3*, int, int) noexcept;

static bench_fn select_backend_fn(const std::string_view backend) noexcept {
  if (backend == "cpu") {
    return multi_exp_cpu;
  }
  if (backend == "gpu") {
    return multi_exp_gpu;
  }
  std::cerr << "invalid backend: " << backend << "\n";
  std::abort();
}

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cerr << "Usage: benchmark <cpu|gpu> <m> <n> <verbose>\n";
    return -1;
  }
  const char* backend = argv[1];
  auto m = std::atoi(argv[2]);
  auto n = std::atoi(argv[3]);
  bool verbose = false;
  if (argc == 5 && std::string_view{argv[4]} == "1") {
    verbose = true;
  }

  auto f = select_backend_fn(backend);

  memmg::managed_array<c21t::element_p3> res(m);

  // invoke f with small values to avoid measuring one-time initialization costs
  f(res.data(), 1, 1);

  auto t1 = std::chrono::steady_clock::now();
  f(res.data(), m, n);
  auto t2 = std::chrono::steady_clock::now();
  double duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0;

  if (verbose) {
    std::cout << "===== result\n";
    for (int i = 0; i < m; ++i) {
      std::cout << res[i] << "\n";
    }
  }

  std::cout << "===== benchmark results\n";
  std::cout << "m = " << m << std::endl;
  std::cout << "n = " << n << std::endl;
  std::cout << "duration (ms): " << duration << "\n";
  auto throughput = static_cast<double>(m) * n / duration;
  std::cout << "throughput: " << throughput << "\n";

  return 0;
}
