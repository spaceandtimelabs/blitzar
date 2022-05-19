extern crate curve25519_dalek;

mod cbindings {
    use cty;

    // pub const SXT_DENSE_SEQUENCE_TYPE: u32 = 1;
    pub const SXT_BACKEND_CPU: cty::c_int = 1;
    pub const SXT_BACKEND_GPU: cty::c_int = 2;

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct sxt_config {
        pub backend: cty::c_int,
    }

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct sxt_dense_sequence_descriptor {
        pub element_nbytes: u8,
        pub n: u64,
        pub data: *const u8,
    }

    #[repr(C)]
    #[derive(Copy, Clone)]
    pub struct sxt_sequence_descriptor {
        pub sequence_type: u8,
        pub __bindgen_anon_1: sxt_sequence_descriptor__bindgen_ty_1,
    }

    #[repr(C)]
    #[derive(Copy, Clone)]
    pub union sxt_sequence_descriptor__bindgen_ty_1 {
        pub dense: sxt_dense_sequence_descriptor,
    }

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct sxt_commitment {
        pub ristretto_bytes: [u8; 32usize],
    }

    extern "C" {
        pub fn sxt_init(config: *const sxt_config) -> cty::c_int;
    }

    extern "C" {
        pub fn sxt_compute_pedersen_commitments(
            commitments: *mut sxt_commitment,
            num_sequences: u32,
            descriptors: *const sxt_sequence_descriptor,
        ) -> cty::c_int;
    }
}

pub enum Backend {
    CPU,
    GPU
}

impl Backend {
    fn value(&self) -> cty::c_int {
        match *self {
            Backend::CPU => cbindings::SXT_BACKEND_CPU,
            Backend::GPU => cbindings::SXT_BACKEND_GPU,
        }
    }
}

pub fn init_backend(curr_backend: Backend) {
    let config: cbindings::sxt_config = cbindings::sxt_config {
        backend: curr_backend.value()
    };

    unsafe {
        let res = cbindings::sxt_init(&config);

        println!("Result init: {}", res);
    };
}

use curve25519_dalek::edwards::CompressedEdwardsY;

pub type Commitment = CompressedEdwardsY;

pub fn compute_commitments(commitments: &mut
        Vec<Commitment>, num_sequences: usize) -> cty::c_int  {
            
    (*commitments) = Vec::with_capacity(num_sequences);

    let mut cbinding_commitments: Vec<cbindings::
            sxt_commitment> = Vec::with_capacity(num_sequences);
    let mut cbinding_descriptors: Vec<cbindings::
            sxt_sequence_descriptor> = Vec::with_capacity(num_sequences);

    unsafe {
        commitments.set_len(num_sequences);
        cbinding_commitments.set_len(num_sequences);
        cbinding_descriptors.set_len(num_sequences);

        let res = cbindings::sxt_compute_pedersen_commitments(
            cbinding_commitments.as_mut_ptr(),
            num_sequences as u32,
            cbinding_descriptors.as_mut_ptr(),
        );

        println!("Result compute: {}", res);
    }

    // copy results back to commitments vector
    for i in 0..num_sequences {
        commitments[i] = CompressedEdwardsY::
                from_slice(&cbinding_commitments[i].ristretto_bytes);
    }

    return 0;
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
