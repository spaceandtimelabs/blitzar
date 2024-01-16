#pragma once

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// bucket_descriptor
//--------------------------------------------------------------------------------------------------
struct bucket_descriptor {
  unsigned num_entries;
  unsigned bucket_index;
  unsigned entry_first;
};
} // namespace sxt::mtxbk
