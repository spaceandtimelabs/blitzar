#pragma once

#include <algorithm>
#include <numeric>
#include <vector>

namespace sxt {
  template <class T>
  static T min(const std::vector<T>& data) {
    return *std::min_element(data.begin(), data.end());
  }

  template <class T>
  static T max(const std::vector<T>& data) {
    return *std::max_element(data.begin(), data.end());
  }

  template <class T>
  static T mean(const std::vector<T>& data) {
    return std::accumulate(data.begin(), data.end(), 0.0) / data.size();
  }

  template <class T>
  static T std_dev(const std::vector<T>& data) {
    const T mean = sxt::mean(data);
    const T variance = std::accumulate(data.begin(), data.end(), 0.0, [mean](T accumulator, T val) {
      return accumulator + (val - mean) * (val - mean);
    }) / data.size();
    return std::sqrt(variance);
  }

  template <class T>
  static T median(const std::vector<T>& data) {
    std::vector<T> sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());
    if (sorted_data.size() % 2 == 0) {
      return (sorted_data[sorted_data.size() / 2 - 1] + sorted_data[sorted_data.size() / 2]) / 2;
    } else {
      return sorted_data[sorted_data.size() / 2];
    }
  }

  template <class T>
  static T gmps(const std::vector<T>& data, unsigned repetitions, unsigned n_elements) {
    const T median = sxt::median(data);
    return 1.0e-9 * repetitions * n_elements / (1.0e-3 * median);
  }
}  // namespace stats
