fn main() {
    if std::env::var_os("PYO3_BUILD_EXTENSION_MODULE").is_some() {
        pyo3_build_config::add_extension_module_link_args();
        return;
    }

    let config = pyo3_build_config::get();
    if let Some(lib_dir) = &config.lib_dir() {
        println!("cargo:rustc-link-search=native={lib_dir}");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_dir}");
    }
    if let Some(lib_name) = &config.lib_name() {
        println!("cargo:rustc-link-lib={lib_name}");
    }
    pyo3_build_config::add_python_framework_link_args();
}
