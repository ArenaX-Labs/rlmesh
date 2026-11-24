use std::collections::HashMap;
use std::time::{Duration, Instant};

pub(crate) const QUIET: Duration = Duration::from_secs(10);

#[derive(Debug, Default)]
pub(crate) struct Quiet {
    last: HashMap<(&'static str, String), (Instant, u64)>,
}

impl Quiet {
    pub(crate) fn admit(&mut self, what: &'static str, subject: &str, now: Instant) -> Option<u64> {
        if self.last.len() > 256 {
            self.last
                .retain(|_, (at, _)| now.saturating_duration_since(*at) < QUIET);
        }
        match self.last.get_mut(&(what, subject.to_owned())) {
            Some((at, held)) if now.saturating_duration_since(*at) < QUIET => {
                *held += 1;
                None
            }
            Some((at, held)) => {
                *at = now;
                Some(std::mem::take(held))
            }
            None => {
                self.last.insert((what, subject.to_owned()), (now, 0));
                Some(0)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_warning_repeats_once_per_quiet_period_and_says_how_many_it_held_back() {
        let mut quiet = Quiet::default();
        let start = Instant::now();
        assert_eq!(quiet.admit("dropped", "eyes", start), Some(0));
        assert_eq!(quiet.admit("dropped", "eyes", start), None);
        assert_eq!(quiet.admit("dropped", "eyes", start + QUIET / 2), None);
        assert_eq!(quiet.admit("dropped", "cam", start), Some(0));
        assert_eq!(quiet.admit("held", "eyes", start), Some(0));
        assert_eq!(quiet.admit("dropped", "eyes", start + QUIET), Some(2));
        assert_eq!(quiet.admit("dropped", "eyes", start + QUIET * 3), Some(0));
    }
}
