extern crate proofs_gpu;

#[cfg(test)]
mod tests {
    #[test]
    fn compute_commitments_works() {
        let config: proofs_gpu::sxt_config = proofs_gpu::sxt_config {
            backend: proofs_gpu::SXT_BACKEND_CPU as i32
        };

        unsafe {
            let ret_init = proofs_gpu::sxt_init(&config);

            assert_eq!(ret_init, 0);
        };
        
        let n1: u64 = 4;
        let num_commitments = 3;
        let n1_num_bytes: u8 = 4;

        let mut data_bytes: [[u32; 4]; 3] = [
            [2000, 7500, 5000, 1500],
            [5000, 0, 400000, 10],
            [2000 + 5000, 7500 + 0, 5000 + 400000, 1500 + 10],
        ];

        let expected_0: [u8; 32] = [
            4,105,58,131,59,69,150,106,
            120,137,32,225,175,244,82,115,
            216,180,206,150,21,250,240,98,
            251,192,146,244,54,169,199,97
        ];

        let expected_1: [u8; 32] = [
            2,254,178,195,198,238,44,156,
            24,29,88,196,37,63,157,50,
            236,159,61,49,153,181,79,126,
            55,188,67,1,228,248,72,51
        ];

        let expected_2: [u8; 32] = [
            30,237,163,234,252,111,45,133,
            235,227,21,117,229,188,88,149,
            240,109,205,90,6,130,199,152,
            5,221,57,231,168,9,141,122
        ];

        let mut cbinding_commitments: Vec<proofs_gpu::
                sxt_commitment> = Vec::with_capacity(num_commitments);
        let mut cbinding_descriptors: Vec<proofs_gpu::
                sxt_sequence_descriptor> = Vec::with_capacity(num_commitments);

        unsafe {
            cbinding_commitments.set_len(num_commitments);
            cbinding_descriptors.set_len(num_commitments);
            
            for i in 0..num_commitments {
                let descriptor = proofs_gpu::sxt_dense_sequence_descriptor {
                    element_nbytes: n1_num_bytes,  // number bytes
                    n: n1,            // number rows
                    data: data_bytes[i].as_mut_ptr() as *const u8   // data pointer
                };

                cbinding_descriptors[i] = proofs_gpu::sxt_sequence_descriptor {
                    sequence_type: proofs_gpu::SXT_DENSE_SEQUENCE_TYPE as u8,
                    __bindgen_anon_1: proofs_gpu::sxt_sequence_descriptor__bindgen_ty_1 {
                        dense: descriptor
                    }
                };
            }

            let ret_compute = proofs_gpu::sxt_compute_pedersen_commitments(
                cbinding_commitments.as_mut_ptr(),
                num_commitments as u32,
                cbinding_descriptors.as_mut_ptr(),
            );

            assert_eq!(ret_compute, 0);
        }

        assert_eq!(cbinding_commitments[0].ristretto_bytes, expected_0);
        assert_eq!(cbinding_commitments[1].ristretto_bytes, expected_1);
        assert_eq!(cbinding_commitments[2].ristretto_bytes, expected_2);
    }
}