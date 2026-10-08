//! Version constraints: `>=8.0,<10.0`, a bare version meaning a minimum.

/// One comparator in a version clause.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Comparator {
    /// `>=X`, or a bare `X`.
    Ge,
    /// `>X`.
    Gt,
    /// `<=X`.
    Le,
    /// `<X`.
    Lt,
    /// `==X` or `=X`.
    Eq,
}

impl Comparator {
    pub(super) fn symbol(self) -> &'static str {
        match self {
            Self::Ge => ">=",
            Self::Gt => ">",
            Self::Le => "<=",
            Self::Lt => "<",
            Self::Eq => "==",
        }
    }
}

/// One clause of a version constraint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Clause {
    pub comparator: Comparator,
    /// Up to three dotted numeric parts.
    pub version: Vec<u64>,
}

pub(super) fn dotted(version: &[u64]) -> String {
    version
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(".")
}

/// Parse a dotted numeric version of one to three parts.
pub(super) fn parse_version(raw: &str) -> Option<Vec<u64>> {
    let parts: Vec<&str> = raw.split('.').collect();
    if parts.len() > 3
        || parts
            .iter()
            .any(|part| part.is_empty() || !part.bytes().all(|b| b.is_ascii_digit()))
    {
        return None;
    }
    parts.iter().map(|part| part.parse().ok()).collect()
}

fn compare(left: &[u64], right: &[u64]) -> std::cmp::Ordering {
    let len = left.len().max(right.len());
    (0..len)
        .map(|i| {
            let a = left.get(i).copied().unwrap_or(0);
            let b = right.get(i).copied().unwrap_or(0);
            a.cmp(&b)
        })
        .find(|ordering| ordering.is_ne())
        .unwrap_or(std::cmp::Ordering::Equal)
}

/// Parse a comma-joined version constraint: each clause an optional `>=`,
/// `>`, `<=`, `<`, `==`, or `=` and a version; a bare version is a minimum.
pub fn parse_constraint(raw: &str) -> Option<Vec<Clause>> {
    if raw.trim().is_empty() {
        return None;
    }
    raw.split(',')
        .map(|clause| {
            let clause = clause.trim();
            let (comparator, rest) = [
                (">=", Comparator::Ge),
                ("<=", Comparator::Le),
                ("==", Comparator::Eq),
                ("=", Comparator::Eq),
                (">", Comparator::Gt),
                ("<", Comparator::Lt),
            ]
            .iter()
            .find_map(|(symbol, comparator)| {
                clause.strip_prefix(symbol).map(|rest| (*comparator, rest))
            })
            .unwrap_or((Comparator::Ge, clause));
            Some(Clause {
                comparator,
                version: parse_version(rest.trim_start())?,
            })
        })
        .collect()
}

/// The lowest version a constraint admits, from its clauses that bound from
/// below (bare, `>=`, `>`, `==`); `None` when none does.
fn constraint_floor(clauses: &[Clause]) -> Option<&[u64]> {
    clauses
        .iter()
        .filter(|clause| !matches!(clause.comparator, Comparator::Lt | Comparator::Le))
        .map(|clause| clause.version.as_slice())
        .max_by(|a, b| compare(a, b))
}

/// Whether a constraint's upper bound already excludes `version`.
pub(super) fn caps_below(clauses: &[Clause], version: &[u64]) -> bool {
    clauses.iter().any(|clause| match clause.comparator {
        Comparator::Lt => compare(&clause.version, version).is_le(),
        Comparator::Le | Comparator::Eq => compare(&clause.version, version).is_lt(),
        _ => false,
    })
}

/// Whether a constraint admits some version below `version`: it has no lower
/// bound (`<13` admits 12.2), or its tightest one sits below `version`.
pub(super) fn admits_below(clauses: &[Clause], version: &[u64]) -> bool {
    constraint_floor(clauses).is_none_or(|floor| compare(floor, version).is_lt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::variant::Requirement;

    #[test]
    fn constraints_parse_like_the_platform() {
        let parse = |raw: &str| {
            parse_constraint(raw).map(|clauses| Requirement::Version(clauses).to_string())
        };
        assert_eq!(parse(">=8.0,<10.0").as_deref(), Some(">=8.0,<10.0"));
        // A bare version is a minimum.
        assert_eq!(parse("12.4").as_deref(), Some(">=12.4"));
        assert_eq!(parse(" >= 12.2 ").as_deref(), Some(">=12.2"));
        assert_eq!(parse("==12.4").as_deref(), Some("==12.4"));
        assert_eq!(parse("=12").as_deref(), Some("==12"));
        assert_eq!(parse(">12,<=12.6.1").as_deref(), Some(">12,<=12.6.1"));
        for bad in [
            "", " ", "12.4.1.2", "~12", ">=", "12,", "abc", "!=12", "12.x",
        ] {
            assert_eq!(parse(bad), None, "{bad:?}");
        }
        let floor = |raw: &str| {
            let clauses = parse_constraint(raw).unwrap();
            constraint_floor(&clauses).map(dotted)
        };
        assert_eq!(floor(">=12.2,>12.4,<13").as_deref(), Some("12.4"));
        assert_eq!(floor("<13"), None);
        assert_eq!(floor("==12.1").as_deref(), Some("12.1"));
    }
}
