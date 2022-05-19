#![allow(unused_imports)]
#![allow(unused_must_use)]

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn build_proofs_gpu_lib() {
    let dst = PathBuf::from(env::var_os("OUT_DIR").unwrap());

    fs::create_dir_all(dst.join("lib"));
    fs::copy("src/proofsgpulib/libproofs-gpu.so", dst.join("lib/libproofs-gpu.so"));

    println!("cargo:root={}", dst.to_str().unwrap());
    println!("cargo:rustc-link-search=native={}", dst.join("lib").to_str().unwrap());
}

fn main() {
    build_proofs_gpu_lib();

    println!("cargo:rustc-link-lib=proofs-gpu");
    println!("cargo:rerun-if-changed=build.rs");
}