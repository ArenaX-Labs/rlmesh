//! Paired reset seeds: one episode's seed from what must stay fixed for a
//! paired comparison, and nothing else.
//!
//! The inputs are the base seed, the seed key (the environment, never the
//! model or a per-launch id) and the episode's global slot within that key. No
//! session id: two models over the same base and environment draw the same
//! episodes, and shards that split one total partition a single slot range, so
//! re-sharding draws the same set. The managed platform derives its seeds with
//! this same function; `conformance/episode_seeds.v1.json` pins it across
//! languages.

/// FNV-1a over (base seed, seed key, slot), masked to `[0, 2**32)` so every
/// seed fits Gymnasium's and NumPy's seed range.
pub fn episode_seed(base_seed: i64, seed_key: &str, slot: u64) -> i64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

    fn update(mut hash: u64, bytes: &[u8]) -> u64 {
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    }

    let mut hash = FNV_OFFSET;
    hash = update(hash, &base_seed.to_le_bytes());
    hash = update(hash, &[0xff]);
    hash = update(hash, seed_key.as_bytes());
    hash = update(hash, &[0xfd]);
    hash = update(hash, &slot.to_le_bytes());
    (hash & u64::from(u32::MAX)) as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn episode_seeds_match_the_cross_language_fixture() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../../../conformance/episode_seeds.v1.json"))
                .unwrap();
        let cases = fixture["cases"].as_array().unwrap();
        assert!(!cases.is_empty());
        for case in cases {
            let base = case["base_seed"].as_i64().unwrap();
            let key = case["seed_key"].as_str().unwrap();
            let slot = case["slot"].as_u64().unwrap();
            assert_eq!(
                episode_seed(base, key, slot),
                case["seed"].as_i64().unwrap(),
                "{case}"
            );
        }
    }

    #[test]
    fn a_different_base_key_or_slot_changes_the_seed() {
        let seed = episode_seed(7, "libero#0", 0);
        assert_ne!(seed, episode_seed(8, "libero#0", 0));
        assert_ne!(seed, episode_seed(7, "libero#1", 0));
        let mut slots: Vec<i64> = (0..64)
            .map(|slot| episode_seed(7, "libero#0", slot))
            .collect();
        slots.sort_unstable();
        slots.dedup();
        assert_eq!(slots.len(), 64);
    }

    #[test]
    fn seeds_fit_gymnasium_range() {
        for slot in 0..64 {
            let seed = episode_seed(i64::MIN, "k", slot);
            assert!(
                (0..=i64::from(u32::MAX)).contains(&seed),
                "seed {seed} out of range"
            );
        }
    }
}
