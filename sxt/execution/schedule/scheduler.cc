#include "sxt/execution/schedule/scheduler.h"

#include <immintrin.h>

#include "sxt/execution/schedule/pollable_event.h"

namespace sxt::xens {
//--------------------------------------------------------------------------------------------------
// run
//--------------------------------------------------------------------------------------------------
void scheduler::run() noexcept {
  // Simple loop to poll and execute ready tasks
  while (true) {
    if (head_ == nullptr) {
      return;
    }
    // Note: this code hasn't been informed by benchmarking yet, and
    // it's likely that it will be adjusted when it does.
    //
    // For now, we emit a pause instruction as that frequently leads to
    // better performance for polling code.
    //
    // See https://stackoverflow.com/q/58424276/4447365
    _mm_pause();

    if (head_->ready()) {
      auto event = std::move(head_);
      head_ = event->release_next();
      event->invoke();
      continue;
    }
    auto event = head_.get();
    while (event->next() != nullptr) {
      auto next = event->next();
      if (next->ready()) {
        next->invoke();
        event->set_next(next->release_next());
      } else {
        event = next;
      }
    }
  }
}

//--------------------------------------------------------------------------------------------------
// schedule
//--------------------------------------------------------------------------------------------------
void scheduler::schedule(std::unique_ptr<pollable_event>&& event) noexcept {
  auto head = std::move(head_);
  head_ = std::move(event);
  head_->set_next(std::move(head));
}

//--------------------------------------------------------------------------------------------------
// get_scheduler
//--------------------------------------------------------------------------------------------------
scheduler& get_scheduler() noexcept {
  static thread_local auto instance = new scheduler{};
  return *instance;
}
} // namespace sxt::xens
