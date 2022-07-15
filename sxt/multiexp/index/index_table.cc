#include "sxt/multiexp/index/index_table.h"

#include <algorithm>
#include <iostream>
#include <iterator>
#include <utility>

namespace sxt::mtxi {
//--------------------------------------------------------------------------------------------------
// constructor
//--------------------------------------------------------------------------------------------------
index_table::index_table(const index_table& other) noexcept
    : index_table{other, other.get_allocator()} {}

index_table::index_table(const index_table& other, allocator_type alloc) noexcept : alloc_{alloc} {
  this->operator=(other);
}

index_table::index_table(index_table&& other) noexcept
    : index_table{std::move(other), other.get_allocator()} {}

index_table::index_table(index_table&& other, allocator_type alloc) noexcept : alloc_{alloc} {
  this->operator=(std::move(other));
}

index_table::index_table(size_t num_rows, size_t max_entries, allocator_type alloc) noexcept
    : alloc_{alloc} {
  this->reshape(num_rows, max_entries);
}

index_table::index_table(std::initializer_list<std::initializer_list<uint64_t>> values,
                         allocator_type alloc) noexcept
    : alloc_{alloc} {
  auto num_rows = values.size();
  size_t entry_count = 0;
  for (auto row : values) {
    entry_count += row.size();
  }
  this->reshape(num_rows, entry_count);

  auto hdr = this->header();
  auto rows_p = values.begin();
  auto entry_data = reinterpret_cast<uint64_t*>(data_ + sizeof(header_type) * num_rows_);
  for (size_t index = 0; index < num_rows; ++index) {
    auto& row = hdr[index];
    auto row_p = *(rows_p + index);
    row = header_type{entry_data, row_p.size()};
    entry_data = std::copy(row_p.begin(), row_p.end(), entry_data);
  }
}

//--------------------------------------------------------------------------------------------------
// destructor
//--------------------------------------------------------------------------------------------------
index_table::~index_table() noexcept { this->reset(); }

//--------------------------------------------------------------------------------------------------
// operator=
//--------------------------------------------------------------------------------------------------
index_table& index_table::operator=(const index_table& rhs) noexcept {
  if (this == &rhs) {
    return *this;
  }
  if (capacity_ < rhs.capacity_) {
    this->reset();
    capacity_ = rhs.capacity_;
    data_ = alloc_.allocate(capacity_);
  }
  num_rows_ = rhs.num_rows_;
  auto hdr = this->header();
  auto hdr_p = rhs.header();

  auto entry_data = this->entry_data();
  for (size_t index = 0; index < num_rows_; ++index) {
    auto& row = hdr[index];
    auto row_p = hdr_p[index];
    row = header_type{entry_data, row_p.size()};
    entry_data = std::copy(row_p.begin(), row_p.end(), entry_data);
  }

  return *this;
}

index_table& index_table::operator=(index_table&& rhs) noexcept {
  if (this->get_allocator() != rhs.get_allocator()) {
    return this->operator=(rhs);
  }

  this->reset();

  num_rows_ = rhs.num_rows_;
  capacity_ = rhs.capacity_;
  data_ = rhs.data_;

  rhs.num_rows_ = 0;
  rhs.capacity_ = 0;
  rhs.data_ = nullptr;

  return *this;
}

//--------------------------------------------------------------------------------------------------
// entry_data
//--------------------------------------------------------------------------------------------------
uint64_t* index_table::entry_data() noexcept {
  return reinterpret_cast<uint64_t*>(data_ + sizeof(header_type) * num_rows_);
}

const uint64_t* index_table::entry_data() const noexcept {
  return reinterpret_cast<const uint64_t*>(data_ + sizeof(header_type) * num_rows_);
}

//--------------------------------------------------------------------------------------------------
// reset
//--------------------------------------------------------------------------------------------------
void index_table::reset() noexcept {
  if (data_ == nullptr) {
    return;
  }
  alloc_.deallocate(data_, capacity_);
  num_rows_ = 0;
  data_ = nullptr;
  capacity_ = 0;
}

//--------------------------------------------------------------------------------------------------
// reshape
//--------------------------------------------------------------------------------------------------
void index_table::reshape(size_t num_rows, size_t max_entries) noexcept {
  auto capacity_p = num_rows * sizeof(basct::span<uint64_t>) + max_entries * sizeof(uint64_t);
  if (capacity_p <= capacity_) {
    num_rows_ = num_rows;
    return;
  }
  this->reset();
  num_rows_ = num_rows;
  capacity_ = capacity_p;
  data_ = alloc_.allocate(capacity_p);
}

//--------------------------------------------------------------------------------------------------
// operator==
//--------------------------------------------------------------------------------------------------
bool operator==(const index_table& lhs, const index_table& rhs) noexcept {
  auto hdr = lhs.header();
  auto hdr_p = rhs.header();
  if (hdr.size() != hdr_p.size()) {
    return false;
  }
  for (size_t index = 0; index < hdr.size(); ++index) {
    auto row = hdr[index];
    auto row_p = hdr_p[index];
    if (row.size() != row_p.size()) {
      return false;
    }
    if (!std::equal(row.begin(), row.end(), row_p.begin())) {
      return false;
    }
  }
  return true;
}

//--------------------------------------------------------------------------------------------------
// operator<<
//--------------------------------------------------------------------------------------------------
std::ostream& operator<<(std::ostream& out, const index_table& tbl) {
  auto hdr = tbl.header();
  out << "{";
  for (auto& row : hdr) {
    out << "{";
    for (auto& entry : row) {
      out << entry;
      if (std::distance(&entry, row.end()) > 1) {
        out << ",";
      }
    }
    out << "}";
    if (std::distance(&row, hdr.end()) > 1) {
      out << ",";
    }
  }
  out << "}";
  return out;
}
} // namespace sxt::mtxi
