//! Runs through native adapters: execution horizons, spec timing, and the
//! instruction and kept context they read.

use super::*;

struct AdapterSteps {
    step: i64,
}
impl Env for AdapterSteps {
    fn observation_space(&self) -> Space {
        crate::spaces::Box::new(-10.0, 10.0, vec![1], Default::default())
            .unwrap()
            .into()
    }
    fn action_space(&self) -> Space {
        crate::spaces::Box::new(-10.0, 10.0, vec![1], Default::default())
            .unwrap()
            .into()
    }
    fn reset(
        &mut self,
        _seed: Option<i64>,
        _options: Option<&InfoDict>,
    ) -> Result<(Value, InfoDict)> {
        self.step = 0;
        Ok((Value::List(vec![Value::Float(2.)]), InfoDict::new()))
    }
    fn step(&mut self, _action: Option<Value>) -> Result<(Value, f64, bool, bool, InfoDict)> {
        self.step += 1;
        Ok((
            Value::List(vec![Value::Float(3.)]),
            1.,
            true,
            false,
            InfoDict::new(),
        ))
    }
    fn render_mode(&self) -> Option<String> {
        Some("rgb_array".into())
    }
    fn render(&mut self) -> Result<Option<Value>> {
        Ok(Some(Value::Int(self.step)))
    }
}
fn adapter_action() -> crate::adapters::Action {
    use crate::adapters::*;
    Action::new([Actuator {
        role: Some(ACTION_GRIPPER.into()),
        dim: 1,
        ..Default::default()
    }])
}
fn adapter_steps() -> crate::adapters::Tagged<AdapterSteps> {
    use crate::adapters::*;
    tag(
        AdapterSteps { step: 0 },
        &EnvTags::new(StateTag::new(GRIPPER_POS), adapter_action()).unwrap(),
        Default::default(),
    )
    .unwrap()
}
struct TextPolicy;
impl crate::Policy for TextPolicy {
    fn predict(&mut self, observation: Value) -> Result<Value> {
        let input = observation.as_dict()?;
        assert_eq!(
            input.get("instruction"),
            Some(&Value::List(vec![Value::Text("override".into())]))
        );
        assert!(input.contains_key("state"));
        Ok(Value::List(vec![Value::Float(0.)]))
    }
}
fn text_model() -> Model {
    use crate::adapters::*;
    Model::new(
        TextPolicy,
        ModelOptions {
            spec: Some(
                ModelSpec::new(
                    InputNode::dict([
                        ("state", State::new(GRIPPER_POS).into()),
                        (
                            "missing_camera",
                            Image {
                                optional: true,
                                height: Some(1),
                                width: Some(1),
                                channels: Some(3),
                                ..Image::new(IMAGE_PRIMARY)
                            }
                            .into(),
                        ),
                        (
                            "instruction",
                            Text {
                                container: TextContainer::List,
                                fill: Some("fallback".into()),
                                ..Text::new(INSTRUCTION)
                            }
                            .into(),
                        ),
                    ]),
                    adapter_action(),
                )
                .unwrap()
                .into(),
            ),
            ..Default::default()
        },
    )
    .unwrap()
}
/// A chunked spec'd policy recording the horizon each chunk is asked at.
struct Paced {
    seen: Arc<Mutex<Vec<u32>>>,
    loaded: Option<u32>,
}
impl crate::Policy for Paced {
    const CORNERS: crate::Corners = crate::Corners::PREDICT.union(crate::Corners::PREDICT_CHUNK);
    fn execution_horizon(&self) -> Option<u32> {
        self.loaded
    }
    fn predict(&mut self, _observation: Value) -> Result<Value> {
        // The runtime takes the single-action corner at horizon 1.
        self.seen.lock().unwrap().push(1);
        Ok(Value::List(vec![Value::Float(0.)]))
    }
    fn predict_chunk(&mut self, _observation: Value, execution_horizon: u32) -> Result<Value> {
        self.seen.lock().unwrap().push(execution_horizon);
        Ok(Value::List(vec![
            Value::List(vec![Value::Float(0.)]);
            execution_horizon as usize
        ]))
    }
}
fn paced(loaded: Option<u32>, timing: Option<u32>) -> (Model, Arc<Mutex<Vec<u32>>>) {
    use crate::adapters::*;
    let seen = Arc::default();
    let spec = ModelSpec::new(State::new(GRIPPER_POS), adapter_action())
        .unwrap()
        .with_timing(Timing { execution: timing })
        .unwrap();
    let model = Model::new(
        Paced {
            seen: Arc::clone(&seen),
            loaded,
        },
        ModelOptions {
            spec: Some(spec.into()),
            ..Default::default()
        },
    )
    .unwrap();
    (model, seen)
}

#[test]
fn a_run_naming_no_horizon_executes_at_the_loaded_one_then_the_spec_timing_then_one() {
    for (loaded, timing, explicit, expected) in [
        (Some(3), Some(2), None, 3),
        (None, Some(2), None, 2),
        (None, None, None, 1),
        (Some(3), Some(2), Some(1), 1),
    ] {
        let (model, seen) = paced(loaded, timing);
        let mut env = adapter_steps();
        model
            .run(
                &mut env,
                RunOptions {
                    execution_horizon: explicit,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(
            *seen.lock().unwrap(),
            vec![expected],
            "{loaded:?} {timing:?} {explicit:?}"
        );
    }
    let (model, seen) = paced(Some(3), None);
    let mut env = adapter_steps();
    let mut binding = model.bind(&mut env, BindOptions::default()).unwrap();
    binding.reset(None, None).unwrap();
    binding.step(None).unwrap();
    drop(binding);
    assert_eq!(*seen.lock().unwrap(), vec![3]);
    let envelope = crate::describe_json(&model, Default::default()).unwrap();
    assert!(envelope.contains(r#""execution_horizon":3"#), "{envelope}");
}

#[test]
fn a_spec_timing_outside_one_to_1024_is_refused() {
    use crate::adapters::*;
    let spec = ModelSpec::new(State::new(GRIPPER_POS), adapter_action()).unwrap();
    for bad in [0, 1025] {
        let error = spec
            .clone()
            .with_timing(Timing {
                execution: Some(bad),
            })
            .unwrap_err();
        assert_eq!(error.class(), ErrorClass::ValueError);
    }
    let timed = spec.with_timing(Timing { execution: Some(8) }).unwrap();
    let back = ModelSpec::from_json(&timed.to_json().unwrap()).unwrap();
    assert_eq!(back.timing(), timed.timing());
    // State and its one-part Concat have the same canonical wire form.
    assert_eq!(back.to_json().unwrap(), timed.to_json().unwrap());
}

#[derive(Default)]
struct ReadHooks {
    context: Option<Arc<dyn RunContext>>,
    reads: usize,
}
impl RunHooks for ReadHooks {
    fn on_run_start(&mut self, context: Arc<dyn RunContext>) -> Result<()> {
        self.context = Some(context);
        Ok(())
    }
    fn on_step(&mut self, event: &StepEvent) -> Result<()> {
        let value = event.read(crate::adapters::GRIPPER_POS)?;
        assert!(matches!(value, Value::Tensor(_)));
        let context = self.context.as_ref().unwrap();
        let mut sources = context.frame_sources()?;
        assert_eq!(sources.render_label.as_deref(), Some("render"));
        assert!(sources.render.as_mut().is_some());
        assert_eq!(sources.render.as_mut().unwrap().call()?, Value::Int(1));
        self.reads += 1;
        Ok(())
    }
    fn on_run_end(&mut self, _result: &RunResult) -> Result<()> {
        let mut sources = self.context.as_ref().unwrap().frame_sources()?;
        assert!(sources.render.as_mut().is_some());
        assert_eq!(sources.render.as_mut().unwrap().call()?, Value::Int(1));
        Ok(())
    }
}

#[test]
fn native_adapter_instruction_and_kept_context_read() {
    let model = text_model();
    let mut env = adapter_steps();
    let mut hooks = ReadHooks::default();
    let result = model
        .run(
            &mut env,
            RunOptions {
                instruction: Some("override".into()),
                hooks: Some(&mut hooks),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(hooks.reads, 1);
    assert!(!result.advisories.is_empty());
    let context = hooks.context.unwrap();
    assert!(context.contract().unwrap().is_some());
    assert!(context.frame_sources().unwrap().render.is_none());
    assert!(
        context
            .read(
                &Value::List(vec![Value::Float(5.)]),
                &crate::adapters::GRIPPER_POS.into()
            )
            .is_ok()
    );
}
