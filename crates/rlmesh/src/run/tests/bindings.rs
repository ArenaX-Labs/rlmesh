//! Bindings: stepping a local env with the model or a given action, reset
//! options, hooks, and which envs a binding closes or refuses.

use super::*;

#[test]
fn a_binding_steps_a_local_env_on_this_thread_with_the_model_or_a_given_action() {
    struct OnThread(Steps, std::thread::ThreadId);
    impl Env for OnThread {
        fn observation_space(&self) -> Space {
            self.0.observation_space()
        }
        fn action_space(&self) -> Space {
            self.0.action_space()
        }
        fn reset(
            &mut self,
            seed: Option<i64>,
            options: Option<&InfoDict>,
        ) -> Result<(Value, InfoDict)> {
            assert_eq!(std::thread::current().id(), self.1);
            self.0.reset(seed, options)
        }
        fn step(&mut self, action: Option<Value>) -> Result<(Value, f64, bool, bool, InfoDict)> {
            assert_eq!(std::thread::current().id(), self.1);
            self.0.step(action)
        }
        fn close(&mut self) -> Result<()> {
            self.0.close()
        }
    }
    let model = ones();
    let mut env = OnThread(Steps::new(), std::thread::current().id());
    let resets = Arc::clone(&env.0.resets);
    let closes = Arc::clone(&env.0.closes);
    let mut binding = model.bind(&mut env, BindOptions::default()).unwrap();
    assert!(binding.done());
    assert_eq!(binding.num_envs(), 1);
    let mut options = InfoDict::new();
    options.insert("level", Value::Int(3));
    let (observation, _) = binding.reset(Some(4), Some(&options)).unwrap();
    assert_eq!(observation, Value::Int(0));
    assert!(!binding.done());
    // The model's action (1), then a given one (0).
    assert_eq!(binding.step(None).unwrap().1, 1.0);
    assert_eq!(binding.step(Some(Value::Int(0))).unwrap().1, 0.0);
    let (observation, _, terminated, _, info) = binding.step(None).unwrap();
    assert_eq!(observation, Value::Int(3));
    assert!(terminated && binding.done());
    assert_eq!(info.get("is_success"), Some(&Value::Bool(true)));
    let error = binding.step(None).unwrap_err();
    assert_eq!(error.class(), ErrorClass::RuntimeError);
    binding.close().unwrap();
    binding.close().unwrap();
    assert_eq!(
        binding.reset(None, None).unwrap_err().class(),
        ErrorClass::RuntimeError
    );
    drop(binding);
    assert_eq!(*resets.lock().unwrap(), vec![(Some(4), Some(options))]);
    // Closing the binding closes neither the borrowed env nor the model.
    assert_eq!(closes.load(Ordering::SeqCst), 0);
    model.run(&mut env, RunOptions::default()).unwrap();
}

#[test]
fn a_policy_that_writes_reset_options_adds_them_to_each_reset_it_starts() {
    struct Levelled;
    impl crate::Policy for Levelled {
        const RESET_OPTIONS: bool = true;
        fn predict(&mut self, _observation: Value) -> Result<Value> {
            Ok(Value::Int(1))
        }
        fn reset_options(&mut self, episode: &Metadata) -> Result<Option<Metadata>> {
            let mut options = Metadata::new();
            options.insert("level", episode.get("seed").cloned().unwrap_or(Value::None));
            assert!(matches!(episode.get("episode_id"), Some(Value::Text(_))));
            Ok(Some(options))
        }
    }
    let model = Model::new(Levelled, ModelOptions::default()).unwrap();
    let mut env = Steps::new();
    let resets = Arc::clone(&env.resets);
    model
        .run(
            &mut env,
            RunOptions {
                seeds: Some(vec![5, 6]),
                ..Default::default()
            },
        )
        .unwrap();
    let levels: Vec<_> = resets
        .lock()
        .unwrap()
        .iter()
        .map(|(_, options)| options.as_ref().and_then(|o| o.get("level").cloned()))
        .collect();
    assert_eq!(levels, vec![Some(Value::Int(5)), Some(Value::Int(6))]);
}

#[test]
fn a_bindings_hooks_see_its_episodes_a_reset_over_one_as_truncated_and_the_run_end_at_close() {
    let model = ones();
    let mut env = Steps::new();
    let mut log = Log::default();
    {
        let mut binding = model
            .bind(
                &mut env,
                BindOptions {
                    hooks: Some(&mut log),
                    ..Default::default()
                },
            )
            .unwrap();
        binding.reset(Some(10), None).unwrap();
        for _ in 0..3 {
            binding.step(None).unwrap();
        }
        // Reset over the second episode after one step: it ends truncated.
        binding.reset(Some(11), None).unwrap();
        binding.step(Some(Value::Int(0))).unwrap();
        binding.reset(None, None).unwrap();
        binding.close().unwrap();
    }
    let context = log.context.take().unwrap();
    assert_eq!(
        context.num_envs().unwrap_err().class(),
        ErrorClass::RuntimeError,
        "the binding's context fails once it closes"
    );
    assert_eq!(
        log.calls,
        vec![
            "run_start contract=true num_envs=1",
            "episode_start 0 Some(10)",
            "step 0 0 obs=0 action=1 reward=1",
            "step 0 1 obs=1 action=1 reward=1",
            "step 0 2 obs=2 action=1 reward=1",
            "episode_end 0 steps=3 reward=3",
            "episode_start 1 Some(11)",
            "step 1 0 obs=0 action=0 reward=0",
            "episode_end 1 steps=1 reward=0",
            "episode_start 2 None",
            "episode_end 2 steps=0 reward=0",
            "run_end episodes=3",
        ]
    );
}

#[test]
fn a_binding_closes_a_factory_built_env_and_a_local_env_only_under_close_env() {
    let model = ones();
    let mut factory = Factory {
        closes: Arc::default(),
        made: 0,
    };
    let closes = Arc::clone(&factory.closes);
    let mut binding = model
        .bind(EnvTarget::factory(&mut factory), BindOptions::default())
        .unwrap();
    binding.reset(None, None).unwrap();
    binding.close().unwrap();
    assert_eq!(closes.load(Ordering::SeqCst), 1);
    let mut env = Steps::new();
    let closes = Arc::clone(&env.closes);
    let options = BindOptions {
        close_env: true,
        ..Default::default()
    };
    drop(model.bind(&mut env, options).unwrap());
    assert_eq!(closes.load(Ordering::SeqCst), 1);
}

#[test]
fn a_binding_reaches_a_served_env_and_refuses_what_it_cannot_drive() {
    let model = ones();
    let server = steps_server();
    let mut binding = model.bind(&server, BindOptions::default()).unwrap();
    binding.reset(Some(1), None).unwrap();
    assert_eq!(binding.step(None).unwrap().0, Value::Int(1));
    binding.close().unwrap();
    // A v1 endpoint, and a served model at one.
    let error = model
        .bind("127.0.0.1:1", BindOptions::default())
        .unwrap_err();
    assert_eq!(error.class(), ErrorClass::ValueError);
    let error =
        crate::RemoteModel::new(crate::RemoteModelOptions::default().address("127.0.0.1:1"))
            .unwrap_err();
    assert_eq!(error.class(), ErrorClass::ValueError);
    server.shutdown().unwrap();
}
