//! Semantic role vocabulary for matching env features to model inputs.
//!
//! Roles are matched as opaque strings by the resolver, but the framework also
//! keeps a [`registry`] over the vocabulary: each domain module ships a `ROLES`
//! table and the registry is their union. The registry is the *mechanism* (a
//! domain-neutral table + a dim law); the domain modules are the *vocabulary*, so
//! a new domain is a new module, not a core change.
//!
//! Width conventions: a registered role may pin its dim via [`registry::DimLaw`],
//! but only as *validation* -- the author always declares `dim=` explicitly and
//! the registry checks it, never supplies it. An unregistered (ad-hoc) role has
//! no dim law and resolves on string agreement alone.

pub mod core;
pub mod manipulation;
pub mod registry;
