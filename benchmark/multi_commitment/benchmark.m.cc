#include <chrono>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <memory>
#include <string_view>

#include "sxt/seqcommit/base/commitment.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/base/num/fast_random_number_generator.h"
#include "benchmark/multi_commitment/multi_commitment_cpu.h"
#include "benchmark/multi_commitment/multi_commitment_gpu.h"

using namespace sxt;

using bench_fn = void(*)(
    memmg::managed_array<sqcb::commitment> &commitments_per_col,
    uint64_t rows, uint64_t cols, uint64_t element_nbytes,
    const memmg::managed_array<uint8_t> &data_table) noexcept;

struct Params {
    int status;
    bool verbose;
    bench_fn func;
    uint64_t cols, rows;
    uint64_t element_nbytes;
    uint64_t table_size;

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

        table_size = cols * rows * element_nbytes;
    }

    void select_backend_fn(const std::string_view backend) noexcept {
        if (backend == "cpu") {
            func = (bench_fn) multi_commitment_cpu;
            return;
        }

        if (backend == "gpu") {
            func = (bench_fn) multi_commitment_gpu;
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

static void print_result(uint64_t cols, memmg::managed_array<sqcb::commitment> &commitments_per_col) {
    std::cout << "===== result\n";

    // print the 32 bytes commitment results of each column
    for (size_t c = 0; c < cols; ++c) {
        std::cout << "commitment " << c << " = " << commitments_per_col[c] << std::endl;
    }
}

static void populate_table(memmg::managed_array<uint8_t> &data_table) {
    basn::fast_random_number_generator generator(data_table.size(), data_table.size());

    for (size_t i = 0; i < data_table.size(); ++i) {
        data_table[i] = (uint8_t) (generator() % 256);
    }
}

static void pre_initialize(bench_fn func) {
    memmg::managed_array<uint8_t> data_table_fake(1); // 1 col, 1 row, 1 bytes per data
    memmg::managed_array<sqcb::commitment> commitments_per_col_fake(1);

    populate_table(data_table_fake);

    func(commitments_per_col_fake, 1, 1, 1, data_table_fake);
}

int main(int argc, char* argv[]) {
    Params p(argc, argv);

    if (p.status != 0) return -1;

    p.trigger_timer();

    // populate data section
    memmg::managed_array<uint8_t> data_table(p.table_size);
    memmg::managed_array<sqcb::commitment> commitments_per_col(p.cols);

    populate_table(data_table);
    
    p.stop_timer();

    double duration_populate = p.elapsed_time();

    // invoke f with small values to avoid measuring one-time initialization costs
    pre_initialize(p.func);

    p.trigger_timer();
    
    p.func(commitments_per_col, p.rows, p.cols, p.element_nbytes, data_table);

    p.stop_timer();

    double duration_compute = p.elapsed_time();
    double data_throughput = p.table_size / duration_compute;

    std::cout << "===== benchmark results\n";
    std::cout << "rows = " << p.rows << std::endl;
    std::cout << "cols = " << p.cols << std::endl;
    std::cout << "element_nbytes = " << p.element_nbytes << std::endl;
    std::cout << "populate duration (s): " << std::fixed << duration_populate << "\n";
    std::cout << "compute duration (s): " << std::fixed << duration_compute << "\n";
    std::cout << "data throughput (bytes / s): " << std::scientific << data_throughput << "\n";

    if (p.verbose) print_result(p.cols, commitments_per_col);
    
    return 0;
}