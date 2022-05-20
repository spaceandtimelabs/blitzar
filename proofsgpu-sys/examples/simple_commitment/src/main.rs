extern crate proofs_gpu;

fn main() {
    let config: proofs_gpu::sxt_config = proofs_gpu::sxt_config {
        backend: proofs_gpu::SXT_BACKEND_CPU as i32
    };

    unsafe {
        let res = proofs_gpu::sxt_init(&config);

        if res != 0 {
            panic!("sxt_init failed");
        }
    };
    
    let num_sequences = 1;
    let mut cbinding_commitments: Vec<proofs_gpu::
            sxt_commitment> = Vec::with_capacity(num_sequences);
    let mut cbinding_descriptors: Vec<proofs_gpu::
            sxt_sequence_descriptor> = Vec::with_capacity(num_sequences);

    unsafe {
        cbinding_commitments.set_len(num_sequences);
        cbinding_descriptors.set_len(num_sequences);

        let n1: u64 = 3;
        let n1_num_bytes: u8 = 1;

        let mut data_bytes_1: [u8; 3] = [1, 2, 3];
        let descriptor1 = proofs_gpu::sxt_dense_sequence_descriptor {
            element_nbytes: n1_num_bytes,  // number bytes
            n: n1,            // number rows
            data: data_bytes_1.as_mut_ptr()   // data pointer
        };

        cbinding_descriptors[0] = proofs_gpu::sxt_sequence_descriptor {
            sequence_type: proofs_gpu::SXT_DENSE_SEQUENCE_TYPE as u8,
            __bindgen_anon_1: proofs_gpu::sxt_sequence_descriptor__bindgen_ty_1 {
                dense: descriptor1
            }
        };

        let res = proofs_gpu::sxt_compute_pedersen_commitments(
            cbinding_commitments.as_mut_ptr(),
            num_sequences as u32,
            cbinding_descriptors.as_mut_ptr(),
        );

        if res != 0 {
            panic!("sxt_compute_pedersen_commitments failed");
        }
    }

    // print commitment results
    for i in 0..num_sequences {
        println!("Res {}: {:?}", i, cbinding_commitments[i].ristretto_bytes);
    }
}