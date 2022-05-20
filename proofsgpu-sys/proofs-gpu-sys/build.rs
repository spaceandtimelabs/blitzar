extern crate bindgen;

use std::env;
use std::fs;
use std::path::Path;
use std::path::PathBuf;

fn run_bindgen() {
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");
    let binding_name = "bindings.rs";
    let include_name = format!("pedersen_capi_v{}.h", VERSION);
    let include_src_path = format!("src/proofsgpulib/{}", include_name);
    let dst = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let include_path = dst.join("include");

    // Generate the rust bindings
    let bindings = bindgen::Builder::default()
        .header(include_src_path)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate bindings");

    if Path::new(include_path.to_str().unwrap()).exists() {
        fs::remove_dir_all(include_path.clone())
            .expect("Error removing include directory");
    }

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    fs::create_dir_all(include_path.clone())
        .expect("Couldn't create include directory");

    bindings
        .write_to_file(include_path.join(binding_name))
        .expect("Couldn't write bindings!");
}

fn build_proofs_gpu_lib() {
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");
    let reduced_lib_name = "proofs-gpu";
    let mut lib_name = format!("lib{}-v{}.so", reduced_lib_name, VERSION);
    let lib_name_local = format!("lib{}.so", reduced_lib_name);

    // verify if the system is supported
    if cfg!(target_os = "linux") {
        lib_name = format!("{}-linux", lib_name);
    } else {
        panic!("You are *not* running linux!");
    }

    // verify if the architecture is supported
    if cfg!(target_arch = "x86") {
        lib_name = format!("{}-x86", lib_name);
    } else if cfg!(target_arch = "x86_64") {
        lib_name = format!("{}-x86_64", lib_name);
    } else {
        panic!("Unsupported architecture");
    }

    let lib_src_path = format!("src/proofsgpulib/{}", lib_name);
    let dst = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let lib_path = dst.join("lib");

    // generate rust bindings to interface with c api
    run_bindgen();

    if Path::new(lib_path.to_str().unwrap()).exists() {
        fs::remove_dir_all(lib_path.clone())
            .expect("Error removing lib directory");
    }

    // copy shared lib to output directory
    fs::create_dir_all(lib_path.clone())
        .expect("Couldn't create lib directory");

    fs::copy(lib_src_path, lib_path.join(lib_name_local))
        .expect("Couldn't find shared library");

    println!("cargo:root={}", dst.to_str().unwrap());
    println!("cargo:rustc-link-search=native={}", lib_path.to_str().unwrap());
    println!("cargo:rustc-link-lib={}", reduced_lib_name);
}

fn main() {
    build_proofs_gpu_lib();

    println!("cargo:rerun-if-changed=build.rs");
}