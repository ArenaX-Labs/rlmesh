//! Cross-surface ABI audit; behavior is covered by each facade module and the
//! C/C++ consumers run by test:cxx.
#[test]
fn old_wrapper_exports_are_removed() {
    let c = include_str!("../include/rlmesh.h");
    let cpp = include_str!("../include/rlmesh.hpp");
    for text in [c, cpp] {
        assert!(!text.contains("rlmesh_ext_legacy_"));
        assert!(!text.contains("RLMeshExtLegacy"));
    }
    assert!(!cpp.contains("RLMESH_TRY"));
    assert!(!cpp.contains("class ValueRef"));
}
