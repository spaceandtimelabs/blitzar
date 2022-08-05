#include <iostream>

#include "sxt/base/profile/callgrind.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/multiexp/curve21/multiproduct_cpu_driver.h"
#include "sxt/multiexp/index/index_table.h"
#include "sxt/multiexp/pippenger_multiprod/multiproduct.h"
#include "sxt/multiexp/test/generate_curve21_elements.h"

using namespace sxt;

int main() {
  mtxi::index_table products{{0, 0, 0},
                             {1, 0, 1},
                             {2, 0, 0, 1},
                             {3, 0, 2},
                             {4, 0, 0, 2},
                             {5, 0, 1, 2},
                             {6, 0, 0, 1, 2},
                             {7, 0, 3},
                             {8, 0, 0, 3},
                             {9, 0, 1, 3},
                             {10, 0, 0, 1, 3},
                             {11, 0, 2, 3},
                             {12, 0, 0, 2, 3},
                             {13, 0, 1, 2, 3},
                             {14, 0, 0, 1, 2, 3},
                             {15, 0, 4},
                             {16, 0, 0, 4},
                             {17, 0, 1, 4},
                             {18, 0, 0, 1, 4},
                             {19, 0, 2, 4},
                             {20, 0, 0, 2, 4},
                             {21, 0, 1, 2, 4},
                             {22, 0, 0, 1, 2, 4},
                             {23, 0, 3, 4},
                             {24, 0, 0, 3, 4},
                             {25, 0, 1, 3, 4},
                             {26, 0, 0, 1, 3, 4},
                             {27, 0, 2, 3, 4},
                             {28, 0, 0, 2, 3, 4},
                             {29, 0, 1, 2, 3, 4},
                             {30, 0, 0, 1, 2, 3, 4},
                             {31, 0, 5},
                             {32, 0, 0, 5},
                             {33, 0, 1, 5},
                             {34, 0, 0, 1, 5},
                             {35, 0, 2, 5},
                             {36, 0, 0, 2, 5},
                             {37, 0, 1, 2, 5},
                             {38, 0, 0, 1, 2, 5},
                             {39, 0, 3, 5},
                             {40, 0, 0, 3, 5},
                             {41, 0, 1, 3, 5},
                             {42, 0, 0, 1, 3, 5},
                             {43, 0, 2, 3, 5},
                             {44, 0, 0, 2, 3, 5},
                             {45, 0, 1, 2, 3, 5},
                             {46, 0, 0, 1, 2, 3, 5},
                             {47, 0, 4, 5},
                             {48, 0, 0, 4, 5},
                             {49, 0, 1, 4, 5},
                             {50, 0, 0, 1, 4, 5},
                             {51, 0, 2, 4, 5},
                             {52, 0, 0, 2, 4, 5},
                             {53, 0, 1, 2, 4, 5},
                             {54, 0, 0, 1, 2, 4, 5},
                             {55, 0, 3, 4, 5},
                             {56, 0, 0, 3, 4, 5},
                             {57, 0, 1, 3, 4, 5},
                             {58, 0, 0, 1, 3, 4, 5},
                             {59, 0, 2, 3, 4, 5},
                             {60, 0, 0, 2, 3, 4, 5},
                             {61, 0, 1, 2, 3, 4, 5},
                             {62, 0, 0, 1, 2, 3, 4, 5}};
  size_t num_entries = 192;
  memmg::managed_array<c21t::element_p3> inout(num_entries);
  std::mt19937 rng{0};
  mtxtst::generate_curve21_elements(inout, rng);
  mtxc21::multiproduct_cpu_driver drv;
  SXT_TOGGLE_COLLECT;
  mtxpmp::compute_multiproduct(inout, products.header(), drv, 6);
  SXT_TOGGLE_COLLECT;
  return 0;
}
