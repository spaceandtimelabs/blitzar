#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string_view>

#include "benchmark/reduce2/reduce_cpu.h"
#include "benchmark/reduce2/reduce_gpu.h"
#include "sxt/curve21/type/element_p3.h"

using namespace sxt;

using bench_fn = void (*)(c21t::element_p3*, int, int) noexcept;

static bench_fn select_backend_fn(const std::string_view backend) noexcept {
  if (backend == "cpu") {
    return reduce_cpu;
  }
  if (backend == "gpu") {
    return reduce_gpu;
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

  std::unique_ptr<c21t::element_p3[]> res{new c21t::element_p3[m]};

  // invoke f with small values to avoid measuring one-time initialization costs
  f(res.get(), 1, 1);

  auto t1 = std::chrono::steady_clock::now();
  f(res.get(), m, n);
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
