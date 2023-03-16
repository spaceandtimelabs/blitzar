#pragma once

namespace sxt::xena {
//--------------------------------------------------------------------------------------------------
// task
//--------------------------------------------------------------------------------------------------
class task {
protected:
  // Task destruction is performed by run_and_dispose with the concrete type
  ~task() noexcept = default;

public:
  virtual void run_and_dispose() noexcept = 0;
};
} // namespace sxt::xena
