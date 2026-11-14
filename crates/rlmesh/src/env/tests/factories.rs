//! Env factories: discriminant binding, the branch table, a fired cancel
//! before prepare, and variant reprs.

use super::*;

/// A factory with one discriminant.
#[derive(Default)]
struct Factory {
    made: Vec<Kwargs>,
}

impl EnvFactory for Factory {
    fn tag_params() -> Vec<TagParam> {
        vec![TagParam {
            name: "mode".into(),
            default_: Value::Text("a".into()),
            choices: vec![Value::Text("a".into()), Value::Text("b".into())],
        }]
    }
    fn reset_options() -> Vec<String> {
        vec!["trial_index".into()]
    }
    fn make(&mut self, kwargs: &Kwargs) -> Result<MadeEnv> {
        self.made.push(kwargs.clone());
        Ok(MadeEnv::env(Counter::new(&Arc::new(Counters::default()))))
    }
}

#[test]
fn make_env_binds_discriminants_and_stamps() {
    let mut factory = Factory::default();
    let MadeEnv::Env(env) = factory.make_env(&Kwargs::new()).unwrap() else {
        panic!("a scalar env")
    };
    assert_eq!(factory.made[0].get("mode"), Some(&Value::Text("a".into())));
    let metadata = env.metadata().unwrap();
    let mut branch = Kwargs::new();
    branch.insert("mode", "a");
    assert_eq!(
        metadata.get(rlmesh_adapters::v1::ENV_BRANCH_METADATA_KEY),
        Some(&Value::Dict(branch))
    );
    assert_eq!(
        metadata.get(crate::ENV_RESET_OPTIONS_KEY),
        Some(&Value::List(vec![Value::Text("trial_index".into())]))
    );

    let mut kwargs = Kwargs::new();
    kwargs.insert("mode", "c");
    let err = factory.make_env(&kwargs).unwrap_err();
    assert_eq!(err.class(), ErrorClass::ValueError);
    assert_eq!(
        err.to_string(),
        "Factory.make() got mode='c', which is not one of the contract discriminant's choices \
         ['a', 'b']"
    );
    assert_eq!(
        factory.made.len(),
        1,
        "make must not run for a refused binding"
    );
}

#[derive(Default)]
struct Unbounded;

impl EnvFactory for Unbounded {
    fn tag_params() -> Vec<TagParam> {
        let choices: Vec<Value> = (0..3).map(Value::Int).collect();
        ["x", "y"]
            .into_iter()
            .map(|name| TagParam {
                name: name.into(),
                default_: Value::Int(0),
                choices: choices.clone(),
            })
            .collect()
    }
    fn make(&mut self, _: &Kwargs) -> Result<MadeEnv> {
        Ok(MadeEnv::env(Counter::new(&Arc::new(Counters::default()))))
    }
}

#[test]
fn the_branch_table_is_checked() {
    let err = Unbounded.make_env(&Kwargs::new()).unwrap_err();
    assert_eq!(err.class(), ErrorClass::TypeError);
    assert!(err.to_string().contains("9 contract branches"), "{err}");
    let branches = super::factory::branches::<Factory>();
    assert_eq!(branches.len(), 2);
    assert_eq!(branches[0].get("mode"), Some(&Value::Text("a".into())));
}

#[test]
fn an_already_fired_cancel_runs_no_prepare() {
    #[derive(Default)]
    struct Prepared(usize);
    impl EnvFactory for Prepared {
        fn prepare(&mut self) -> Result<()> {
            self.0 += 1;
            Ok(())
        }
        fn make(&mut self, _: &Kwargs) -> Result<MadeEnv> {
            Ok(MadeEnv::env(Counter::new(&Arc::new(Counters::default()))))
        }
    }
    let cancel = crate::ext::CancelToken::new();
    cancel.cancel();
    let mut factory = Prepared::default();
    let err = factory
        .serve(
            ":0/env",
            EnvFactoryServeOptions {
                cancel: Some(cancel.clone()),
                ..Default::default()
            },
        )
        .unwrap_err();
    assert_eq!(err.class(), ErrorClass::KeyboardInterrupt);
    assert_eq!(factory.0, 0);

    // A cancel fired while serving drains, closes and fails with the class.
    let canceller = cancel.clone();
    let thread = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(200));
        canceller.cancel();
    });
    let err = factory
        .serve(
            ":0/env",
            EnvFactoryServeOptions {
                cancel: Some(cancel),
                ..Default::default()
            },
        )
        .unwrap_err();
    thread.join().unwrap();
    assert_eq!(err.class(), ErrorClass::KeyboardInterrupt);
    assert_eq!(factory.0, 1);
}

#[test]
fn variants_print_python_s_repr() {
    let mut params = Kwargs::new();
    params.insert("task", 1);
    let mut metadata = Kwargs::new();
    metadata.insert("name", "Pick it's");
    let variant = Variant::new("a/b", params, VariantOptions { metadata });
    assert_eq!(
        variant.to_string(),
        "Variant(id='a/b', params={'task': 1}, metadata={'name': \"Pick it's\"})"
    );
    assert_eq!(variant.clone(), variant);
}
