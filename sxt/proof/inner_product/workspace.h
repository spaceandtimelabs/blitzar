#pragma once

namespace sxt::s25t {
struct element;
}

namespace sxt::prfip {
//--------------------------------------------------------------------------------------------------
// workspace
//--------------------------------------------------------------------------------------------------
/**
 * Proving an inner product proceeds over multiple rounds. This abstraction allows a backend for the
 * computational steps to persist data between prover rounds.
 */
class workspace {
public:
  virtual ~workspace() noexcept = default;

  /**
   * On the final round of an inner product proof for the product of vectors <a, b> where b is
   * known, the prover will have repeatedly folded the vector a down to a single element a'. This
   * function provides an accessor to the a' value.
   */
  virtual void ap_value(s25t::element& value) const noexcept = 0;
};
} // namespace sxt::prfip
