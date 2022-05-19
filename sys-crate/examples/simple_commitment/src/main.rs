extern crate proofs_gpu;

fn main() {
    proofs_gpu::init_backend(proofs_gpu::Backend::GPU);
    
    let num_sequences: usize = 5;
    let mut commitments: Vec<proofs_gpu::Commitment> = Vec::new();

    proofs_gpu::compute_commitments(&mut commitments, num_sequences);

    println!("{:?}", commitments);
}