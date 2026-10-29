//! Named projections: lossy fits a binding applies only when allowed, and
//! records by name (in a v2 binding's table, on a v1 resolved adapter).

#[allow(unused_imports)]
use crate::prelude::*;

use core::fmt;

use crate::spec::policy::Allow;

/// A named projection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Projection {
    /// Discard a model output no actuator consumes.
    DropAction,
    /// Read the leading elements of a wider env state field.
    SelectState,
    /// Feed a model's camera input from the env's only camera, under another
    /// role (a wrist input reading a third-person view).
    RebindCamera,
    /// Reverse a gripper value whose polarity the two sides declare
    /// differently.
    FlipGripper,
    /// Rescale a delta command whose unit the two sides declare differently.
    ConvertUnits,
}

impl Projection {
    /// Every projection.
    pub const ALL: [Self; 5] = [
        Self::DropAction,
        Self::SelectState,
        Self::RebindCamera,
        Self::FlipGripper,
        Self::ConvertUnits,
    ];

    /// Its name, in an `allow` list and in the record.
    pub fn name(self) -> &'static str {
        match self {
            Self::DropAction => "drop_action",
            Self::SelectState => "select_state",
            Self::RebindCamera => "rebind_camera",
            Self::FlipGripper => "flip_gripper",
            Self::ConvertUnits => "convert_units",
        }
    }

    /// The projection `name` names.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|p| p.name() == name)
    }

    /// The v2 consent that permits it, when v2 has one: a v2 policy names
    /// each camera by path, so it never rebinds one.
    pub fn allow(self) -> Option<Allow> {
        match self {
            Self::DropAction => Some(Allow::Drop),
            Self::SelectState => Some(Allow::Select),
            Self::RebindCamera => None,
            Self::FlipGripper => Some(Allow::Rescale),
            Self::ConvertUnits => Some(Allow::Units),
        }
    }
}

impl fmt::Display for Projection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}
