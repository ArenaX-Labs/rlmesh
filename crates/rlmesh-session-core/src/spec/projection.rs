//! Named projections: lossy fits a binding applies only when allowed, and
//! records by name (in a v2 binding's table, on a v1 resolved adapter).

use std::fmt;

use crate::spec::policy::Allow;

/// A named projection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Projection {
    /// Discard a model output no actuator consumes.
    DropAction,
}

impl Projection {
    /// Every projection.
    pub const ALL: [Self; 1] = [Self::DropAction];

    /// Its name, in an `allow` list and in the record.
    pub fn name(self) -> &'static str {
        match self {
            Self::DropAction => "drop_action",
        }
    }

    /// The projection `name` names.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|p| p.name() == name)
    }

    /// The v2 consent that permits it.
    pub fn allow(self) -> Allow {
        match self {
            Self::DropAction => Allow::Drop,
        }
    }
}

impl fmt::Display for Projection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}
