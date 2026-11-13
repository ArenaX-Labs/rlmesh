//! Spaces this release accepts, checked by the released rlmesh-spaces
//! 0.1.0 validator: what a runner bridging a 0.2 env to a 0.1 model meets
//! when it hands the 0.1 side the env's contract. Each case says whether the
//! released reader takes it; a 0.2-only form is a documented break.

use rlmesh_spaces as current;
use rlmesh_spaces_0_1 as released;

fn text(min_length: i64) -> (current::SpaceSpec, released::SpaceSpec) {
    (
        current::SpaceSpec {
            shape: Vec::new(),
            dtype: current::DType::Uint8,
            spec: Some(current::SpaceKind::Text(current::TextSpec {
                min_length,
                max_length: 64,
                charset: String::new(),
            })),
        },
        released::SpaceSpec {
            shape: Vec::new(),
            dtype: released::DType::Uint8,
            spec: Some(released::SpaceKind::Text(released::TextSpec {
                min_length,
                max_length: 64,
                charset: String::new(),
            })),
        },
    )
}

/// Gymnasium's `Text(min_length=0)` (an empty string allowed) is valid in
/// 0.2 and refused by the 0.1 validator, as the compatibility doc says.
#[test]
fn an_empty_text_minimum_is_0_2_only() {
    let (ours, theirs) = text(0);
    current::validate_space(&ours).expect("0.2 accepts min_length 0");
    let refused = released::validate_space(&theirs).expect_err("0.1 refuses it");
    assert!(
        refused.to_string().contains("min_length must be > 0"),
        "{refused}"
    );

    let (ours, theirs) = text(1);
    current::validate_space(&ours).expect("0.2 accepts min_length 1");
    released::validate_space(&theirs).expect("0.1 accepts min_length 1");
}
