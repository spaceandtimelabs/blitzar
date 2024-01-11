#pragma once

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// bucket_descriptor
//--------------------------------------------------------------------------------------------------
struct bucket_descriptor {
  unsigned bucket_index;
  unsigned entry_first;
  unsigned entry_last;
};
} // namespace sxt::mtxbk
