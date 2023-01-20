#include "sxt/multiexp/pippenger_multiprod/prune.h"

#include <algorithm>
#include <iterator>
#include <utility>
#include <vector>

#include "sxt/base/error/assert.h"
#include "sxt/base/iterator/counting_iterator.h"
#include "sxt/multiexp/pippenger_multiprod/active_count.h"

namespace sxt::mtxpmp {
//--------------------------------------------------------------------------------------------------
// prune_rows
//--------------------------------------------------------------------------------------------------
void prune_rows(basct::span<basct::span<uint64_t>> rows, std::vector<uint64_t>& markers,
                size_t& num_inactive_outputs, size_t& num_inactive_inputs) noexcept {
  // get the count for each entry
  std::vector<size_t> v1(markers.size());
  std::vector<size_t> max_active_counts(markers.size());
  for (size_t row_index = num_inactive_outputs; row_index < rows.size(); ++row_index) {
    auto& row = rows[row_index];
    auto num_inactive_entries = row[1];
    auto num_active_entries = row.size() - 2 - num_inactive_entries;
    SXT_DEBUG_ASSERT(num_active_entries > 0);
    for (size_t entry_index = 2 + num_inactive_entries; entry_index < row.size(); ++entry_index) {
      auto entry = row[entry_index];
      ++v1[entry];
      SXT_DEBUG_ASSERT(entry < max_active_counts.size());
      max_active_counts[entry] = std::max(max_active_counts[entry], num_active_entries);
    }
  }

  // partition the entries to separate out the isolated ones
  std::vector<uint64_t> v2(markers.size());
  auto [sep, sep_p] = std::partition_copy(basit::counting_iterator<uint64_t>{0},
                                          basit::counting_iterator<uint64_t>{markers.size()},
                                          v2.begin(), v2.rbegin(), [&](size_t index) noexcept {
                                            SXT_DEBUG_ASSERT(v1[index] > 0);
                                            return v1[index] == 1 || max_active_counts[index] == 1;
                                          });
  auto deactivation_count = static_cast<size_t>(std::distance(v2.begin(), sep));
  if (deactivation_count == 0) {
    return;
  }
  size_t counter = 0;
  for (auto iter = v2.begin(); iter != sep; ++iter) {
    v1[*iter] = markers.size() + num_inactive_inputs + counter++;
  }
  counter = 0;
  for (auto iter = v2.rbegin(); iter != sep_p; ++iter) {
    v1[*iter] = counter++;
  }

  // reindex and update the inactive counts
  for (size_t row_index = num_inactive_outputs; row_index < rows.size(); ++row_index) {
    auto& row = rows[row_index];
    auto& num_inactive_entries = row[1];
    std::tie(sep, sep_p) = std::partition_copy(
        row.begin() + 2 + num_inactive_entries, row.end(), v2.begin(), v2.rbegin(),
        [&](uint64_t index) noexcept { return v1[index] >= markers.size(); });
    for (auto iter = v2.begin(); iter != sep; ++iter) {
      row[2 + num_inactive_entries++] = v1[*iter] - markers.size();
    }
    auto active_iter = row.data() + 2 + num_inactive_entries;
    for (auto iter = v2.rbegin(); iter != sep_p; ++iter) {
      *active_iter++ = v1[*iter];
    }
    if (2 + num_inactive_entries == row.size()) {
      std::swap(row, rows[num_inactive_outputs]);
      ++num_inactive_outputs;
    }
  }

  // update the markers
  for (size_t index = 0; index < markers.size(); ++index) {
    auto index_p = v1[index];
    if (index_p >= markers.size()) {
      v2[index_p - markers.size() - num_inactive_inputs] = markers[index];
    } else {
      v2[index_p + deactivation_count] = markers[index];
    }
  }
  markers.swap(v2);
  num_inactive_inputs += deactivation_count;
}
} // namespace sxt::mtxpmp
