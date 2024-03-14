/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023-present Space and Time Labs, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <algorithm>
#include <numeric>
#include <vector>

namespace sxt {
//--------------------------------------------------------------------------------------------------
// min
//--------------------------------------------------------------------------------------------------
template <class T> static T min(const std::vector<T>& data) {
  return *std::min_element(data.begin(), data.end());
}

//--------------------------------------------------------------------------------------------------
// max
//--------------------------------------------------------------------------------------------------
template <class T> static T max(const std::vector<T>& data) {
  return *std::max_element(data.begin(), data.end());
}

//--------------------------------------------------------------------------------------------------
// mean
//--------------------------------------------------------------------------------------------------
template <class T> static T mean(const std::vector<T>& data) {
  return std::accumulate(data.begin(), data.end(), 0.0) / data.size();
}

//--------------------------------------------------------------------------------------------------
// std_dev
//--------------------------------------------------------------------------------------------------
template <class T> static T std_dev(const std::vector<T>& data) {
  const T mean = sxt::mean(data);
  const T variance = std::accumulate(data.begin(), data.end(), 0.0,
                                     [mean](T accumulator, T val) {
                                       return accumulator + (val - mean) * (val - mean);
                                     }) /
                     (data.size() - 1);
  return std::sqrt(variance);
}

//--------------------------------------------------------------------------------------------------
// median
//--------------------------------------------------------------------------------------------------
template <class T> static T median(const std::vector<T>& data) {
  std::vector<T> sorted_data = data;

  auto n = sorted_data.size() / 2;
  std::nth_element(sorted_data.begin(), sorted_data.begin() + n, sorted_data.end());

  if (sorted_data.size() % 2 == 0) {
    T max_of_lower_half = *std::max_element(sorted_data.begin(), sorted_data.begin() + n);
    return (max_of_lower_half + sorted_data[n]) / 2;
  } else {
    return sorted_data[n];
  }
}

//--------------------------------------------------------------------------------------------------
// gmps
//--------------------------------------------------------------------------------------------------
/**
 * Giga operations per second. The data vector must be in milliseconds.
 */
template <class T>
static T gmps(const std::vector<T>& data, unsigned repetitions, unsigned n_elements) {
  const T median = sxt::median(data);
  return 1.0e-9 * repetitions * n_elements / (1.0e-3 * median);
}
} // namespace sxt
