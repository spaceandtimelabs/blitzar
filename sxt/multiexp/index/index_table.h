#pragma once

#include <cstdint>
#include <initializer_list>
#include <iosfwd>

#include "sxt/base/container/span.h"
#include "sxt/base/type/polymorphic_allocator.h"

namespace sxt::mtxi {
//--------------------------------------------------------------------------------------------------
// index_table
//--------------------------------------------------------------------------------------------------
class index_table {
 public:
   using allocator_type = bast::polymorphic_allocator;
   using header_type = basct::span<uint64_t>;
   using const_header_type = basct::span<const uint64_t>;

   // constructors
   index_table() noexcept = default;

   explicit index_table(allocator_type alloc) noexcept : alloc_{alloc} {}

   index_table(const index_table& other) noexcept;

   index_table(const index_table& other, allocator_type alloc) noexcept;

   index_table(index_table&& other) noexcept;

   index_table(index_table&& other, allocator_type alloc) noexcept;

   index_table(size_t num_rows, size_t max_entries,
               allocator_type alloc = {}) noexcept;

   index_table(std::initializer_list<std::initializer_list<uint64_t>> values,
               allocator_type alloc = {}) noexcept;

   // destructor
   ~index_table() noexcept;

   // assignment
   index_table& operator=(const index_table& rhs) noexcept;

   index_table& operator=(index_table&& rhs) noexcept;

   // accessors
   allocator_type get_allocator() const noexcept { return alloc_; }

   uint64_t* entry_data() noexcept;
   const uint64_t* entry_data() const noexcept;

   size_t num_rows() const noexcept { return num_rows_; }

   // methods
   bool empty() const noexcept { return num_rows_ == 0; }

   void reset() noexcept;

   basct::span<header_type> header() noexcept {
     return basct::span<header_type>{reinterpret_cast<header_type*>(data_),
                                     num_rows_};
   }

   basct::span<const const_header_type> header() const noexcept {
     return basct::span<const const_header_type>{
         reinterpret_cast<const_header_type*>(data_), num_rows_};
   }

   basct::span<const const_header_type> cheader() const noexcept {
     return this->header();
   }

   void reshape(size_t num_rows, size_t max_entries) noexcept;

  private:
   allocator_type alloc_;
   size_t num_rows_{0};
   size_t capacity_{0};
   std::byte* data_ = nullptr;
};

//--------------------------------------------------------------------------------------------------
// operator==
//--------------------------------------------------------------------------------------------------
bool operator==(const index_table& lhs, const index_table& rhs) noexcept;

//--------------------------------------------------------------------------------------------------
// operator!=
//--------------------------------------------------------------------------------------------------
inline bool operator!=(const index_table& lhs, const index_table& rhs) noexcept {
  return !(lhs == rhs);
}

//--------------------------------------------------------------------------------------------------
// operator<<
//--------------------------------------------------------------------------------------------------
std::ostream& operator<<(std::ostream& out, const index_table& tbl);
} // namesapce sxt::mtxi
