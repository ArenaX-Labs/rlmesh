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
