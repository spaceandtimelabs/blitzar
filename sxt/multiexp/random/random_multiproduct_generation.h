#pragma once

#include <random>
#include <cstddef>
#include <memory_resource>

#include "sxt/base/container/span.h"
#include "sxt/memory/management/managed_array_fwd.h"

namespace sxt::mtxi { class index_table; }

namespace sxt::mtxrn {
struct random_multiproduct_descriptor;

//--------------------------------------------------------------------------------------------------
// generate_random_multiproduct
//--------------------------------------------------------------------------------------------------
void generate_random_multiproduct(
    mtxi::index_table& products, size_t& num_inputs, size_t& num_entries,
    std::mt19937& rng,
    const random_multiproduct_descriptor& descriptor) noexcept;
}  // namespace sxt::mtxrn
