//! Units (§5.1): SI and REP-103 by default, with the alternatives a source
//! may declare per field.
//!
//! Every field of a core kind has one [`Quantity`], so every field has one
//! unit. Angles and angular velocities convert automatically (rad ↔ deg, in
//! floating point); the other alternatives convert only under `allow:
//! units`; tactile and custom units compare by equality only.

#[allow(unused_imports)]
use crate::prelude::*;

use core::f64::consts::PI;
use core::fmt;

/// What a field measures.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Quantity {
    /// `rad` (SI), `deg`.
    Angle,
    /// `rad/s` (SI), `deg/s`.
    AngularVelocity,
    /// `m` (SI), `mm`.
    Length,
    /// `m/s` (SI), `mm/s`.
    LinearVelocity,
    /// `m/s^2` (SI), `g` (9.80665).
    Acceleration,
    /// `N`.
    Force,
    /// `N*m`.
    Torque,
    /// `Pa` (SI), `kPa`.
    Pressure,
    /// A gripper opening: `normalized` (0 closed … 1 open), or `m` when the
    /// part declares `width_m`.
    Gripper,
    /// Taxel values: `normalized`, `N` or `Pa`, by equality only.
    Tactile,
    /// A custom kind's field: any unit string, by equality only.
    Custom,
    /// No unit: images, rotations, unit vectors, text, flags.
    None,
}

/// How one unit relates to its quantity's SI unit.
#[derive(Clone, Copy, Debug, PartialEq)]
struct UnitDef {
    name: &'static str,
    quantity: Quantity,
    /// `si = value × to_si`.
    to_si: f64,
}

/// The linear units: quantity, name and factor to SI. The first of each
/// quantity is its SI unit.
const LINEAR: [UnitDef; 14] = [
    UnitDef {
        name: "rad",
        quantity: Quantity::Angle,
        to_si: 1.0,
    },
    UnitDef {
        name: "deg",
        quantity: Quantity::Angle,
        to_si: PI / 180.0,
    },
    UnitDef {
        name: "rad/s",
        quantity: Quantity::AngularVelocity,
        to_si: 1.0,
    },
    UnitDef {
        name: "deg/s",
        quantity: Quantity::AngularVelocity,
        to_si: PI / 180.0,
    },
    UnitDef {
        name: "m",
        quantity: Quantity::Length,
        to_si: 1.0,
    },
    UnitDef {
        name: "mm",
        quantity: Quantity::Length,
        to_si: 1e-3,
    },
    UnitDef {
        name: "m/s",
        quantity: Quantity::LinearVelocity,
        to_si: 1.0,
    },
    UnitDef {
        name: "mm/s",
        quantity: Quantity::LinearVelocity,
        to_si: 1e-3,
    },
    UnitDef {
        name: "m/s^2",
        quantity: Quantity::Acceleration,
        to_si: 1.0,
    },
    UnitDef {
        name: "g",
        quantity: Quantity::Acceleration,
        to_si: 9.806_65,
    },
    UnitDef {
        name: "N",
        quantity: Quantity::Force,
        to_si: 1.0,
    },
    UnitDef {
        name: "N*m",
        quantity: Quantity::Torque,
        to_si: 1.0,
    },
    UnitDef {
        name: "Pa",
        quantity: Quantity::Pressure,
        to_si: 1.0,
    },
    UnitDef {
        name: "kPa",
        quantity: Quantity::Pressure,
        to_si: 1e3,
    },
];

/// The gripper's units.
pub const NORMALIZED: &str = "normalized";

impl Quantity {
    /// The SI (default) unit; empty for [`Quantity::None`] and custom
    /// fields.
    pub fn si(self) -> &'static str {
        match self {
            Self::Gripper | Self::Tactile => NORMALIZED,
            Self::Custom | Self::None => "",
            q => LINEAR
                .iter()
                .find(|u| u.quantity == q)
                .map_or("", |u| u.name),
        }
    }

    /// The units a source may declare for a field of this quantity.
    pub fn alternatives(self) -> Vec<&'static str> {
        match self {
            Self::Gripper => vec![NORMALIZED, "m"],
            Self::Tactile => vec![NORMALIZED, "N", "Pa"],
            Self::Custom | Self::None => Vec::new(),
            q => LINEAR
                .iter()
                .filter(|u| u.quantity == q)
                .map(|u| u.name)
                .collect(),
        }
    }

    /// Whether `unit` is a unit of this quantity (any string for custom
    /// fields, which compare by equality).
    pub fn accepts(self, unit: &str) -> bool {
        match self {
            Self::Custom => !unit.is_empty() && unit.len() <= 32,
            Self::None => unit.is_empty(),
            q => q.alternatives().contains(&unit),
        }
    }

    /// Whether conversions between this quantity's units are automatic
    /// (§7.1: angles in floating point).
    pub fn converts_automatically(self) -> bool {
        matches!(self, Self::Angle | Self::AngularVelocity)
    }
}

impl fmt::Display for Quantity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Angle => "angle",
            Self::AngularVelocity => "angular velocity",
            Self::Length => "length",
            Self::LinearVelocity => "linear velocity",
            Self::Acceleration => "acceleration",
            Self::Force => "force",
            Self::Torque => "torque",
            Self::Pressure => "pressure",
            Self::Gripper => "gripper opening",
            Self::Tactile => "tactile",
            Self::Custom => "custom",
            Self::None => "unitless",
        })
    }
}

/// How a value in one unit becomes a value in another.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UnitChange {
    /// The same unit.
    Same,
    /// `to = from × factor`, automatic (angles).
    Auto(f64),
    /// `to = from × factor`, needs `allow: units`.
    Allowed(f64),
    /// Gripper metres to normalised over `width_m = (low, high)`, or back;
    /// needs `allow: units`.
    GripperWidth {
        /// Metres to normalised (`true`) or normalised to metres.
        to_normalized: bool,
    },
    /// No conversion exists (another quantity, or units compared by
    /// equality).
    Incompatible,
}

/// The change from `from` to `to`, both units of `quantity`.
pub fn change(quantity: Quantity, from: &str, to: &str) -> UnitChange {
    if from == to {
        return UnitChange::Same;
    }
    match quantity {
        Quantity::Gripper => match (from, to) {
            ("m", NORMALIZED) => UnitChange::GripperWidth {
                to_normalized: true,
            },
            (NORMALIZED, "m") => UnitChange::GripperWidth {
                to_normalized: false,
            },
            _ => UnitChange::Incompatible,
        },
        Quantity::Tactile | Quantity::Custom | Quantity::None => UnitChange::Incompatible,
        q => {
            let def = |name: &str| LINEAR.iter().find(|u| u.quantity == q && u.name == name);
            match (def(from), def(to)) {
                (Some(a), Some(b)) => {
                    let factor = a.to_si / b.to_si;
                    if q.converts_automatically() {
                        UnitChange::Auto(factor)
                    } else {
                        UnitChange::Allowed(factor)
                    }
                }
                _ => UnitChange::Incompatible,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn angles_convert_automatically_and_lengths_need_allow() {
        let UnitChange::Auto(f) = change(Quantity::Angle, "deg", "rad") else {
            panic!("deg -> rad is automatic");
        };
        assert!((f * 180.0 - PI).abs() < 1e-12);
        assert_eq!(
            change(Quantity::Length, "mm", "m"),
            UnitChange::Allowed(1e-3)
        );
        assert_eq!(
            change(Quantity::Acceleration, "g", "m/s^2"),
            UnitChange::Allowed(9.806_65)
        );
        assert_eq!(
            change(Quantity::Tactile, "N", NORMALIZED),
            UnitChange::Incompatible
        );
        assert_eq!(change(Quantity::Custom, "raw", "raw"), UnitChange::Same);
        assert_eq!(
            change(Quantity::Gripper, "m", NORMALIZED),
            UnitChange::GripperWidth {
                to_normalized: true
            }
        );
        assert_eq!(
            change(Quantity::Force, "N", "kPa"),
            UnitChange::Incompatible
        );
    }

    #[test]
    fn each_quantity_has_its_si_unit_first() {
        assert_eq!(Quantity::Angle.si(), "rad");
        assert_eq!(Quantity::Torque.si(), "N*m");
        assert_eq!(Quantity::Gripper.si(), NORMALIZED);
        assert_eq!(Quantity::Pressure.alternatives(), ["Pa", "kPa"]);
        assert!(Quantity::Custom.accepts("raw") && !Quantity::None.accepts("m"));
    }
}
