#include <chrono>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <memory>
#include <string>
#include <cmath>
#include <random>
#include <cassert>
#include <algorithm>
#include <string_view>

#include "sxt/base/container/span.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/seqcommit/base/commitment.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/seqcommit/base/indexed_exponent_sequence.h"
#include "sxt/seqcommit/generator/base_element.h"
#include "sxt/curve21/ristretto/byte_conversion.h"
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
    memmg::managed_array<sqcb::indexed_exponent_sequence> &data_cols, 
    memmg::managed_array<sqcb::commitment> &generators) {
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint8_t> distribution(0, UINT8_MAX);
    
    for (size_t i = 0; i < rows; ++i) {
        c21t::element_p3 g_i;
        sqcgn::compute_base_element(g_i, i);
        c21rs::to_bytes(generators[i].data(), g_i);
    }
    
    for (size_t i = 0; i < data_table.size(); ++i) {
        data_table[i] = distribution(gen);
    }

    for (size_t c = 0; c < cols; ++c) {
        auto &data_col = data_cols[c];

        data_col.exponent_sequence.n = rows;
        data_col.exponent_sequence.element_nbytes = element_nbytes;
        data_col.exponent_sequence.data = data_table.data() + c * rows * element_nbytes;
    }
}

//--------------------------------------------------------------------------------------------------
// main
//--------------------------------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    params p(argc, argv);

    if (p.status != 0) return -1;

    long long int table_size = (p.cols * p.rows * p.element_nbytes) / 1024 / 1024;

    std::cout << "===== benchmark results" << std::endl;
    std::cout << "backend : " << p.backend_str << std::endl;
    std::cout << "rows : " << p.rows << std::endl;
    std::cout << "cols : " << p.cols << std::endl;
    std::cout << "element_nbytes : " << p.element_nbytes << std::endl;
    std::cout << "table_size (MB) : " << table_size << std::endl;
    std::cout << "num_exponentations : " << (p.cols * p.rows) << std::endl;

    // populate data section
    memmg::managed_array<sqcb::commitment> generators(p.rows);
    memmg::managed_array<sqcb::indexed_exponent_sequence> data_cols(p.cols);
    memmg::managed_array<sqcb::commitment> commitments_per_col(p.cols);
    memmg::managed_array<uint8_t> data_table(p.rows * p.cols * p.element_nbytes);
    basct::span<sqcb::commitment> commitments(commitments_per_col.data(), p.cols);
    basct::cspan<sqcb::indexed_exponent_sequence> value_sequences(data_cols.data(), p.cols);

    populate_table(p.cols, p.rows, p.element_nbytes, data_table, data_cols, generators);

    for (auto use_generators : {true, false}) {
        int NUM_SAMPLES = 15;
        std::vector<double> durations;
        double mean_duration_compute = 0;
        
        for (int i = 0; i < NUM_SAMPLES; ++i) {
            if (use_generators) {
                // populate generators
                basct::span<sqcb::commitment> span_generators(generators.data(), p.rows);

                p.trigger_timer();
                p.backend->compute_commitments(commitments, value_sequences, span_generators);
                p.stop_timer();
            } else {
                basct::span<sqcb::commitment> empty_generators;

                p.trigger_timer();
                p.backend->compute_commitments(commitments, value_sequences, empty_generators);
                p.stop_timer();
            }

            double duration_compute = p.elapsed_time();

            durations.push_back(duration_compute);
            mean_duration_compute += duration_compute / NUM_SAMPLES;
        }

        double std_deviation = 0;
        
        for (int i = 0; i < NUM_SAMPLES; ++i) {
            std_deviation += pow(durations[i] - mean_duration_compute, 2.);
        }

        std_deviation = sqrt(std_deviation / NUM_SAMPLES);

        double data_throughput = p.rows * p.cols / mean_duration_compute;

        std::cout << "use generators : " << (use_generators ? "yes" : "no") << std::endl;
        std::cout << "compute duration (s) : " << std::fixed << mean_duration_compute << std::endl;
        std::cout << "compute std deviation (s) : " << std::fixed << std_deviation << std::endl;
        std::cout << "throughput (exponentiations / s) : " << std::scientific << data_throughput << std::endl;

        if (p.verbose) print_result(p.cols, commitments_per_col);
    }
    
    return 0;
}
