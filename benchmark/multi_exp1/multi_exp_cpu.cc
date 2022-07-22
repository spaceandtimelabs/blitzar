#include "benchmark/multi_exp1/multi_exp_cpu.h"

#include "benchmark/multi_exp1/multiply_add.h"
#include "sxt/curve21/constant/zero.h"
#include "sxt/curve21/type/element_p3.h"

namespace sxt {
//--------------------------------------------------------------------------------------------------
// multi_exp_cpu
//--------------------------------------------------------------------------------------------------
void multi_exp_cpu(c21t::element_p3* res, int m, int n) noexcept {
  for (int mi = 0; mi < m; ++mi) {
    auto& res_mi = res[mi];
    res_mi = c21cn::zero_p3_v;
    for (int i = 0; i < n; ++i) {
      multiply_add(res_mi, mi, i);
    }
  }
}
} // namespace sxt
