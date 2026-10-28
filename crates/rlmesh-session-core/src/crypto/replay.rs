//! A receiver's replay window for one (`key_id`, `kind`) flow (§8.2).

/// The highest `seq` seen and a 64-entry bitmap of the 64 `seq`s below it.
/// A `seq` that was seen, or is more than 64 below the highest, is a replay.
/// `seq` 0 is never sent (counters start at 1), so it is always one.
///
/// [`check`](Self::check) before authenticating and [`insert`](Self::insert)
/// after, so a forged message never moves the window.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReplayWindow {
    high: u64,
    /// Bit `i` set: `high - 1 - i` was seen.
    below: u64,
}

/// How many `seq`s below the highest the window remembers.
pub const WINDOW: u64 = 64;

/// The `seq` was seen before, or is older than the window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("replayed: seq {0} was seen or is older than the window")]
pub struct Replayed(pub u64);

impl ReplayWindow {
    /// An empty window.
    pub const fn new() -> Self {
        Self { high: 0, below: 0 }
    }

    /// The highest `seq` accepted (0 before the first).
    pub fn high(&self) -> u64 {
        self.high
    }

    /// Whether `seq` would be accepted. Changes nothing.
    ///
    /// # Errors
    /// [`Replayed`] if `seq` was seen or is older than the window.
    pub fn check(&self, seq: u64) -> Result<(), Replayed> {
        if seq > self.high {
            return Ok(());
        }
        let back = self.high - seq;
        if seq == 0 || back == 0 || back > WINDOW || self.below & (1 << (back - 1)) != 0 {
            return Err(Replayed(seq));
        }
        Ok(())
    }

    /// Record an authenticated `seq`.
    ///
    /// # Errors
    /// [`Replayed`] if `seq` was seen or is older than the window; the
    /// window is unchanged then.
    pub fn insert(&mut self, seq: u64) -> Result<(), Replayed> {
        self.check(seq)?;
        if seq > self.high {
            let shift = seq - self.high;
            // The old high becomes bit `shift - 1`; everything slides with it.
            self.below = if shift > WINDOW {
                0
            } else {
                let moved = if shift == WINDOW {
                    0
                } else {
                    self.below << shift
                };
                if self.high == 0 {
                    moved
                } else {
                    moved | 1 << (shift - 1)
                }
            };
            self.high = seq;
        } else {
            self.below |= 1 << (self.high - seq - 1);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_in_order_and_rejects_duplicates() {
        let mut w = ReplayWindow::new();
        for seq in 1..=200 {
            w.insert(seq).unwrap();
            assert_eq!(w.insert(seq), Err(Replayed(seq)));
        }
        assert_eq!(w.check(0), Err(Replayed(0)));
    }

    #[test]
    fn seq_zero_is_always_a_replay() {
        let w = ReplayWindow::new();
        assert_eq!(w.check(0), Err(Replayed(0)));
    }

    #[test]
    fn reordering_inside_the_window_is_absorbed_once() {
        let mut w = ReplayWindow::new();
        w.insert(100).unwrap();
        for seq in [99, 37, 36, 50, 98] {
            w.insert(seq).unwrap();
            assert_eq!(w.check(seq), Err(Replayed(seq)), "{seq} twice");
        }
        // 36 is exactly 64 below 100: the oldest the window holds.
        assert_eq!(w.check(35), Err(Replayed(35)), "older than the window");
        assert!(w.check(38).is_ok());
        assert_eq!(w.high(), 100);
    }

    #[test]
    fn the_window_slides_with_the_high_water() {
        let mut w = ReplayWindow::new();
        w.insert(10).unwrap();
        w.insert(8).unwrap();
        w.insert(70).unwrap();
        // 10 and 8 are still remembered, 9 is still open.
        assert_eq!(w.check(10), Err(Replayed(10)));
        assert_eq!(w.check(8), Err(Replayed(8)));
        assert!(w.check(9).is_ok());
        w.insert(74).unwrap();
        // 74 - 64 = 10 is the oldest left; 9 and 8 fell out.
        assert_eq!(w.check(10), Err(Replayed(10)));
        assert_eq!(w.check(9), Err(Replayed(9)));
        assert!(w.check(11).is_ok());
        assert_eq!(w.check(70), Err(Replayed(70)));
        // A jump of exactly the window, and past it, keeps only the old high.
        w.insert(74 + 64).unwrap();
        assert_eq!(w.check(74), Err(Replayed(74)));
        assert!(w.check(75).is_ok());
        w.insert(1000).unwrap();
        assert!(w.check(999).is_ok());
        assert_eq!(w.check(1000 - 65), Err(Replayed(935)));
    }

    #[test]
    fn a_failed_insert_changes_nothing() {
        let mut w = ReplayWindow::new();
        w.insert(5).unwrap();
        let before = w;
        assert!(w.insert(5).is_err());
        assert_eq!(w, before);
    }

    #[test]
    fn handles_the_top_of_the_range() {
        let mut w = ReplayWindow::new();
        w.insert(u64::MAX - 1).unwrap();
        w.insert(u64::MAX).unwrap();
        assert_eq!(w.check(u64::MAX), Err(Replayed(u64::MAX)));
        assert_eq!(w.check(u64::MAX - 1), Err(Replayed(u64::MAX - 1)));
        assert!(w.check(u64::MAX - 2).is_ok());
    }
}
