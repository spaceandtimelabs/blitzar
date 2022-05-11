#pragma once

#include "sxt/base/device/cuda_utility.h"

namespace sxt::basdv {

class stream {
public:
    // constructor
    stream() noexcept {
        basdv::handle_cuda_error(cudaStreamCreate(&stream_));
    }

    ~stream() noexcept {
        basdv::handle_cuda_error(cudaStreamDestroy(stream_));
    }

    /* Prohibits from receiving another stream */
    stream(const stream& other) = delete;

    stream(stream&& other) = delete;

    stream& operator=(stream&&) = delete;

    stream& operator=(const stream&) = delete;

    cudaStream_t getStream() {
        return stream_;
    }

private:
    cudaStream_t stream_;
};

}  // namespace sxt::basdv