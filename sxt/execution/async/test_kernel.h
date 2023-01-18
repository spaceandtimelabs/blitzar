#pragma once

namespace sxt::xenb {
class stream;
}

namespace sxt::xena {
//--------------------------------------------------------------------------------------------------
// add_for_testing
//--------------------------------------------------------------------------------------------------
void add_for_testing(double* c, const xenb::stream& stream, double* a, double* b, int n) noexcept;
} // namespace sxt::xena
