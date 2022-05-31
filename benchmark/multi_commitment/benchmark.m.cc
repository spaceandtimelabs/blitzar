#include <chrono>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <memory>
#include <string>
#include <string_view>

#include "sxt/base/container/span.h"
#include "sxt/seqcommit/base/commitment.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/multiexp/base/exponent_sequence.h"
#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/seqcommit/cbindings/pedersen_backend.h"
#include "sxt/seqcommit/cbindings/pedersen_cpu_backend.h"
#include "sxt/seqcommit/cbindings/pedersen_gpu_backend.h"

using namespace sxt;

struct params {
    int status;
    bool verbose;
    uint64_t cols, rows;
    std::string backend_str;
    uint64_t element_nbytes;
    std::unique_ptr<sqccb::pedersen_backend> backend;
    
    std::chrono::steady_clock::time_point begin_time;
    std::chrono::steady_clock::time_point end_time;
    
    params(int argc, char* argv[]) {
        status = 0;

        if (argc < 5) {
            std::cerr << "Usage: benchmark <cpu|gpu> <rows> <cols> <element_nbytes> <verbose>\n";
            status = -1;
        }

        select_backend_fn(argv[1]);

        verbose = false;
        cols = std::atoi(argv[2]);
        rows = std::atoi(argv[3]);
        element_nbytes = std::atoi(argv[4]);
        
        if (argc == 6 && std::string_view{argv[5]} == "1") {
            verbose = true;
        }

        if (cols <= 0 || rows <= 0 || element_nbytes > 32 || element_nbytes < 0) {
            std::cerr << "Restriction: 1 <= cols, 1 <= rows, 1 <= element_nbytes <= 32\n";
            status = -1;
        }
    }

    void select_backend_fn(const std::string_view backend_view) noexcept {
        if (backend_view == "cpu") {
            backend_str = "cpu";
            backend = std::make_unique<sqccb::pedersen_cpu_backend>();
            return;
        }

        if (backend_view == "gpu") {
            backend_str = "gpu";
            backend = std::make_unique<sqccb::pedersen_gpu_backend>();
            return;
        }

        std::cerr << "invalid backend: " << backend_view << "\n";

        status = -1;
    }

    void trigger_timer() {
        begin_time = std::chrono::steady_clock::now();
    }

    void stop_timer() {
        end_time = std::chrono::steady_clock::now();
    }

    double elapsed_time() {
        return std::chrono::duration_cast<std::chrono::microseconds>(end_time - begin_time).count() / 1e6;
    }
};

//--------------------------------------------------------------------------------------------------
// print_result
//--------------------------------------------------------------------------------------------------
static void print_result(uint64_t cols, memmg::managed_array<sqcb::commitment> &commitments_per_col) {
    std::cout << "===== result\n";

    // print the 32 bytes commitment results of each column
    for (size_t c = 0; c < cols; ++c) {
        std::cout << "commitment " << c << " = " << commitments_per_col[c] << std::endl;
    }
}

//--------------------------------------------------------------------------------------------------
// populate_table
//--------------------------------------------------------------------------------------------------
static void populate_table(
    uint64_t cols, uint64_t rows, uint8_t element_nbytes,
    memmg::managed_array<uint8_t> &data_table, 
    memmg::managed_array<mtxb::exponent_sequence> &data_cols) {

    basn::fast_random_number_generator generator(data_table.size(), data_table.size());

    for (size_t i = 0; i < data_table.size(); ++i) {
        data_table[i] = (uint8_t) (generator() % 256);
    }

    for (size_t c = 0; c < cols; ++c) {
        auto &data_col = data_cols[c];

        data_col.n = rows;
        data_col.element_nbytes = element_nbytes;
        data_col.data = (data_table.data() + c * rows * element_nbytes);
    }
}

//--------------------------------------------------------------------------------------------------
// main
//--------------------------------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    params p(argc, argv);

    if (p.status != 0) return -1;

    p.trigger_timer();

    // populate data section
    memmg::managed_array<uint8_t> data_table(p.rows * p.cols * p.element_nbytes);
    memmg::managed_array<sqcb::commitment> commitments_per_col(p.cols);
    memmg::managed_array<mtxb::exponent_sequence> data_cols(p.cols);
    basct::span<sqcb::commitment> commitments(commitments_per_col.data(), p.cols);
    basct::cspan<mtxb::exponent_sequence> value_sequences(data_cols.data(), p.cols);

    populate_table(p.cols, p.rows, p.element_nbytes, data_table, data_cols);
    
    p.stop_timer();

    double duration_populate = p.elapsed_time();

    p.trigger_timer();
    
    p.backend->compute_commitments(commitments, value_sequences);

    p.stop_timer();

    long long int table_size = (p.cols * p.rows * p.element_nbytes) / 1024 / 1024;
    double duration_compute = p.elapsed_time();
    double data_throughput = p.rows * p.cols / duration_compute;

    std::cout << "===== benchmark results" << std::endl;
    std::cout << "rows : " << p.rows << std::endl;
    std::cout << "cols : " << p.cols << std::endl;
    std::cout << "element_nbytes : " << p.element_nbytes << std::endl;
    std::cout << "table_size (MB) : " << table_size << std::endl;
    std::cout << "num_exponentations : " << (p.cols * p.rows) << std::endl;
    std::cout << "populate duration (s) : " << std::fixed << duration_populate << std::endl;
    std::cout << "compute duration (s) : " << std::fixed << duration_compute << std::endl;
    std::cout << "throughput (exponentiations / s): " << std::scientific << data_throughput << std::endl;
    std::cout << "backend : " << p.backend_str << std::endl;

    if (p.verbose) print_result(p.cols, commitments_per_col);
    
    return 0;
}
