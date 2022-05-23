use std::env;
use std::fs;
use std::path::Path;
use std::path::PathBuf;

fn build_proofs_gpu_lib() {
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");
    let reduced_lib_name = "proofs-gpu";
    let mut lib_name = format!("lib{}-v{}.so", reduced_lib_name, VERSION);
    let lib_name_local = format!("lib{}.so", reduced_lib_name);

    // verify if the system is supported
    if cfg!(target_os = "linux") {
        lib_name = format!("{}-linux", lib_name);
    } else {
        panic!("Unsupported OS");
    }

    // verify if the architecture is supported
    if cfg!(target_arch = "x86") {
        lib_name = format!("{}-x86", lib_name);
    } else if cfg!(target_arch = "x86_64") {
        lib_name = format!("{}-x86_64", lib_name);
    } else {
        panic!("Unsupported architecture");
    }

    let lib_src_path = format!("{}", lib_name);
    let dst = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let lib_path = dst.join("lib");

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
