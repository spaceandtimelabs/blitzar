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
#include "sxt/ristretto/type/compressed_element.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/seqcommit/base/indexed_exponent_sequence.h"
#include "sxt/seqcommit/generator/base_element.h"
#include "sxt/ristretto/base/byte_conversion.h"
#include "sxt/seqcommit/backend/pedersen_backend.h"
#include "sxt/seqcommit/backend/naive_cpu_backend.h"
#include "sxt/seqcommit/backend/naive_gpu_backend.h"
#include "sxt/seqcommit/backend/pippenger_cpu_backend.h"

using namespace sxt;

struct params {
    int status;
    bool verbose;
    int num_samples = 1;
    uint64_t commitments, commitment_length;
    std::string backend_str;
    uint64_t element_nbytes;
    bool is_boolean;
    std::unique_ptr<sqcbck::pedersen_backend> backend;
    
    std::chrono::steady_clock::time_point begin_time;
    std::chrono::steady_clock::time_point end_time;
    
    params(int argc, char* argv[]) {
        status = 0;

        if (argc < 7) {
            std::cerr << "Usage: benchmark <naive-cpu|naive-gpu|pip-cpu>" <<
                         "<commitment_length> <commitments> <element_nbytes>" <<
                         "<verbose> <num_samples>\n";
            status = -1;
        }

        select_backend_fn(argv[1]);

        verbose = is_boolean = false;
        commitments = std::atoi(argv[2]);
        commitment_length = std::atoi(argv[3]);
        element_nbytes = std::atoi(argv[4]);

        if (element_nbytes == 0) {
            is_boolean = true;
            element_nbytes = 1;
        }
        
        if (std::string_view{argv[5]} == "1") {
            verbose = true;
        }

        num_samples = std::atoi(argv[6]);

        if (commitments <= 0 || commitment_length <= 0 || element_nbytes > 32) {
            std::cerr << "Restriction: 1 <= commitments, " << 
                "1 <= commitment_length, 1 <= element_nbytes <= 32\n";
            status = -1;
        }
    }

    void select_backend_fn(const std::string_view backend_view) noexcept {
        if (backend_view == "naive-cpu") {
            backend_str = "naive-cpu";
            backend = std::make_unique<sqcbck::naive_cpu_backend>();
            return;
        }

        if (backend_view == "naive-gpu") {
            backend_str = "naive-gpu";
            backend = std::make_unique<sqcbck::naive_gpu_backend>();
            return;
        }

        if (backend_view == "pip-cpu") {
            backend_str = "pip-cpu";
            backend = std::make_unique<sqcbck::pippenger_cpu_backend>();
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
        return std::chrono::duration_cast
            <std::chrono::microseconds>(end_time - begin_time).count() / 1e6;
    }
};

//--------------------------------------------------------------------------------------------------
// print_result
//--------------------------------------------------------------------------------------------------
static void print_result(uint64_t commitments,
    memmg::managed_array<rstt::compressed_element> &commitments_per_sequence) {

    std::cout << "===== result\n";

    // print the 32 bytes commitment results of each sequence
    for (size_t c = 0; c < commitments; ++c) {
        std::cout << "commitment " << c << " = "
            << commitments_per_sequence[c] << std::endl;
    }
}

//--------------------------------------------------------------------------------------------------
// populate_table
//--------------------------------------------------------------------------------------------------
static void populate_table(
    bool is_boolean,
    uint64_t commitments, uint64_t commitment_length, uint8_t element_nbytes,
    memmg::managed_array<uint8_t> &data_table, 
    memmg::managed_array<sqcb::indexed_exponent_sequence> &data_commitments, 
    memmg::managed_array<rstt::compressed_element> &generators) {
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint8_t> distribution;

    if (is_boolean) {
        distribution = std::uniform_int_distribution<uint8_t>(0, 1);
    } else {
        distribution = std::uniform_int_distribution<uint8_t>(0, UINT8_MAX);
    }
    
    for (size_t i = 0; i < commitment_length; ++i) {
        c21t::element_p3 g_i;
        sqcgn::compute_base_element(g_i, i);
        rstb::to_bytes(generators[i].data(), g_i);
    }
    
    for (size_t i = 0; i < data_table.size(); ++i) {
        data_table[i] = distribution(gen);
    }

    for (size_t c = 0; c < commitments; ++c) {
        auto &data_sequence = data_commitments[c];

        data_sequence.indices = nullptr;
        data_sequence.exponent_sequence.n = commitment_length;
        data_sequence.exponent_sequence.element_nbytes = element_nbytes;
        data_sequence.exponent_sequence.data =
            data_table.data() + c * commitment_length * element_nbytes;
    }
}

//--------------------------------------------------------------------------------------------------
// main
//--------------------------------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    params p(argc, argv);

    if (p.status != 0) return -1;

    long long int table_size = (p.commitments *
        p.commitment_length * p.element_nbytes) / 1024 / 1024;

    std::cout << "===== benchmark results" << std::endl;
    std::cout << "backend : " << p.backend_str << std::endl;
    std::cout << "commitment length : " << p.commitment_length << std::endl;
    std::cout << "number of commitments : " << p.commitments << std::endl;
    std::cout << "element_nbytes : " << p.element_nbytes << std::endl;
    std::cout << "is boolean : " << p.is_boolean << std::endl;
    std::cout << "table_size (MB) : " << table_size << std::endl;
    std::cout << "num_exponentations : " << (p.commitments * p.commitment_length) << std::endl;
    std::cout << "********************************************" << std::endl;

    // populate data section
    memmg::managed_array<rstt::compressed_element> generators(p.commitment_length);
    memmg::managed_array<sqcb::indexed_exponent_sequence> data_commitments(p.commitments);
    memmg::managed_array<rstt::compressed_element> commitments_per_sequence(p.commitments);
    memmg::managed_array<uint8_t> data_table(p.commitment_length * p.commitments * p.element_nbytes);

    basct::span<rstt::compressed_element> commitments(commitments_per_sequence.data(), p.commitments);
    basct::cspan<sqcb::indexed_exponent_sequence> value_sequences(data_commitments.data(), p.commitments);

    populate_table(
        p.is_boolean,
        p.commitments,
        p.commitment_length,
        p.element_nbytes,
        data_table,
        data_commitments,
        generators
    );

    for (auto use_generators : {true, false}) {
        std::vector<double> durations;
        double mean_duration_compute = 0;
        
        for (int i = 0; i < p.num_samples; ++i) {
            if (use_generators) {
                // populate generators
                basct::span<rstt::compressed_element>
                    span_generators(generators.data(), p.commitment_length);

                p.trigger_timer();
                p.backend->compute_commitments(commitments, value_sequences, span_generators);
                p.stop_timer();
            } else {
                basct::span<rstt::compressed_element> empty_generators;

                p.trigger_timer();
                p.backend->compute_commitments(commitments, value_sequences, empty_generators);
                p.stop_timer();
            }

            double duration_compute = p.elapsed_time();

            durations.push_back(duration_compute);
            mean_duration_compute += duration_compute / p.num_samples;
        }

        double std_deviation = 0;
        
        for (int i = 0; i < p.num_samples; ++i) {
            std_deviation += pow(durations[i] - mean_duration_compute, 2.);
        }

        std_deviation = sqrt(std_deviation / p.num_samples);

        double data_throughput = p.commitment_length * p.commitments / mean_duration_compute;

        std::cout << "pre-computed generators : " << (use_generators ? "yes" : "no") << std::endl;
        std::cout << "compute duration (s) : " << std::fixed << mean_duration_compute << std::endl;
        std::cout << "compute std deviation (s) : " << std::fixed << std_deviation << std::endl;
        std::cout << "throughput (exponentiations / s) : " << std::scientific << data_throughput << std::endl;

        if (p.verbose) print_result(p.commitments, commitments_per_sequence);

        std::cout << "********************************************" << std::endl;
    }
    
    return 0;
}
