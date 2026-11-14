//! Cross-release checks of the env tags an env publishes under
//! `rlmesh.adapters.v1.env_tags`: a model built on an older release resolves
//! its adapter against a newer env with the reader it shipped, so every tag
//! shape this release writes must still pass that reader. The checks are
//! this crate's tests; it has no API.
