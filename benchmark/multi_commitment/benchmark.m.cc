#include <chrono>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <memory>
#include <string_view>

#include "sxt/base/container/span.h"
#include "sxt/seqcommit/base/commitment.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/multiexp/base/exponent_sequence.h"
#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/seqcommit/naive/commitment_computation_cpu.h"
#include "sxt/seqcommit/naive/commitment_computation_gpu.h"

using namespace sxt;

using bench_fn = void(*)(
    basct::span<sqcb::commitment> commitments,
    basct::cspan<mtxb::exponent_sequence> value_sequences) noexcept;

struct Params {
    int status;
    bool verbose;
    bench_fn func;
    uint64_t cols, rows;
    uint64_t element_nbytes;

    std::chrono::steady_clock::time_point begin_time;
    std::chrono::steady_clock::time_point end_time;
    
    Params(int argc, char* argv[]) {
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

        if (cols < 0 || rows < 0 || element_nbytes > 32 || element_nbytes < 0) {
            std::cerr << "Restriction: 1 <= cols, 1 <= rows, 1 <= element_nbytes <= 32\n";
            status = -1;
        }
    }

    void select_backend_fn(const std::string_view backend) noexcept {
        if (backend == "cpu") {
            func = (bench_fn) sqcnv::compute_commitments;
            return;
        }

        if (backend == "gpu") {
            func = (bench_fn) sqcnv::compute_commitments_gpu;
            return;
        }

        std::cerr << "invalid backend: " << backend << "\n";

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
// pre_initialize
//--------------------------------------------------------------------------------------------------
static void pre_initialize(bench_fn func) {
    memmg::managed_array<uint8_t> data_table_fake(1); // 1 col, 1 row, 1 bytes per data
    memmg::managed_array<sqcb::commitment> commitments_per_col_fake(1);
    memmg::managed_array<mtxb::exponent_sequence> data_cols_fake(1);
    basct::span<sqcb::commitment> commitments_fake(commitments_per_col_fake.data(), 1);
    basct::cspan<mtxb::exponent_sequence> value_sequences_fake(data_cols_fake.data(), 1);

    populate_table(1, 1, 1, data_table_fake, data_cols_fake);

    func(commitments_fake, value_sequences_fake);
}

//--------------------------------------------------------------------------------------------------
// main
//--------------------------------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    Params p(argc, argv);

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

    // invoke f with small values to avoid measuring one-time initialization costs
    pre_initialize(p.func);

    p.trigger_timer();
    
    p.func(commitments, value_sequences);

    p.stop_timer();

    double duration_compute = p.elapsed_time();
    double data_throughput = p.rows * p.cols / duration_compute;

    std::cout << "===== benchmark results\n";
    std::cout << "rows = " << p.rows << std::endl;
    std::cout << "cols = " << p.cols << std::endl;
    std::cout << "element_nbytes = " << p.element_nbytes << std::endl;
    std::cout << "populate duration (s): " << std::fixed << duration_populate << "\n";
    std::cout << "compute duration (s): " << std::fixed << duration_compute << "\n";
    std::cout << "throughput (exponentiations / s): " << std::scientific << data_throughput << "\n";

    if (p.verbose) print_result(p.cols, commitments_per_col);
    
    return 0;
}