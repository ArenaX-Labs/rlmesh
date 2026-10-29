// Compile every portable options setter and keep C++20 aggregate syntax available.
#include <rlmesh.hpp>

void options_setters() {
  static_assert(std::is_aggregate_v<rlmesh::ToDlpackOptions>);
  {
    rlmesh::ToDlpackOptions options;
    options.with_dl_device(options.dl_device).with_copy(options.copy);
    std::move(options).with_dl_device(options.dl_device).with_copy(options.copy);
  }
  static_assert(std::is_aggregate_v<rlmesh::spaces::BoxOptions>);
  {
    rlmesh::spaces::BoxOptions options;
    options.with_dtype(options.dtype);
    std::move(options).with_dtype(options.dtype);
  }
  static_assert(std::is_aggregate_v<rlmesh::spaces::DiscreteOptions>);
  {
    rlmesh::spaces::DiscreteOptions options;
    options.with_start(options.start).with_dtype(options.dtype);
    std::move(options).with_start(options.start).with_dtype(options.dtype);
  }
  static_assert(std::is_aggregate_v<rlmesh::spaces::MultiDiscreteOptions>);
  {
    rlmesh::spaces::MultiDiscreteOptions options;
    options.with_dtype(options.dtype);
    std::move(options).with_dtype(options.dtype);
  }
  static_assert(std::is_aggregate_v<rlmesh::spaces::TextOptions>);
  {
    rlmesh::spaces::TextOptions options;
    options.with_min_length(options.min_length).with_charset(options.charset);
    std::move(options).with_min_length(options.min_length).with_charset(options.charset);
  }
  static_assert(std::is_aggregate_v<rlmesh::adapters::PreviousOptions>);
  {
    rlmesh::adapters::PreviousOptions options;
    options.with_part(options.part).with_fill(options.fill);
    std::move(options).with_part(options.part).with_fill(options.fill);
  }
  static_assert(std::is_aggregate_v<rlmesh::adapters::ResolveOptions>);
  {
    rlmesh::adapters::ResolveOptions options;
    options.with_trust_entrypoints(options.trust_entrypoints)
        .with_check_inverse(options.check_inverse);
    std::move(options)
        .with_trust_entrypoints(options.trust_entrypoints)
        .with_check_inverse(options.check_inverse);
  }
  static_assert(std::is_aggregate_v<rlmesh::ServeOptions>);
  {
    rlmesh::ServeOptions options;
    options.with_allow_remote_shutdown(options.allow_remote_shutdown)
        .with_idle_timeout(options.idle_timeout)
        .with_drain_timeout(options.drain_timeout)
        .with_close_timeout(options.close_timeout)
        .with_compress_responses(options.compress_responses)
        .with_workflow_edition(options.workflow_edition);
    std::move(options)
        .with_allow_remote_shutdown(options.allow_remote_shutdown)
        .with_idle_timeout(options.idle_timeout)
        .with_drain_timeout(options.drain_timeout)
        .with_close_timeout(options.close_timeout)
        .with_compress_responses(options.compress_responses)
        .with_workflow_edition(options.workflow_edition);
  }
  static_assert(std::is_aggregate_v<rlmesh::ext::EnvServerOptions>);
  {
    rlmesh::ext::EnvServerOptions options;
    options.with_token(options.token)
        .with_peer_info_extra(options.peer_info_extra)
        .with_publish_describe(options.publish_describe);
    std::move(options)
        .with_token(options.token)
        .with_peer_info_extra(options.peer_info_extra)
        .with_publish_describe(options.publish_describe);
  }
  static_assert(std::is_aggregate_v<rlmesh::ext::ModelServeOptions>);
  {
    rlmesh::ext::ModelServeOptions options;
    options.with_token(options.token)
        .with_predict_concurrency(options.predict_concurrency)
        .with_no_signal_handlers(options.no_signal_handlers);
    std::move(options)
        .with_token(options.token)
        .with_predict_concurrency(options.predict_concurrency)
        .with_no_signal_handlers(options.no_signal_handlers);
  }
  static_assert(std::is_aggregate_v<rlmesh::EnvServerOptions>);
  {
    rlmesh::EnvServerOptions options;
    options.with_address(options.address)
        .with_host(options.host)
        .with_port(options.port)
        .with_path(options.path)
        .with_transport(options.transport)
        .with_options(options.options)
        .with_tags(options.tags)
        .with_close_env_on_shutdown(options.close_env_on_shutdown)
        .with_ext(options.ext);
    std::move(options)
        .with_address(options.address)
        .with_host(options.host)
        .with_port(options.port)
        .with_path(options.path)
        .with_transport(options.transport)
        .with_options(options.options)
        .with_tags(options.tags)
        .with_close_env_on_shutdown(options.close_env_on_shutdown)
        .with_ext(options.ext);
  }
  static_assert(std::is_aggregate_v<rlmesh::adapters::TagOptions>);
  {
    rlmesh::adapters::TagOptions options;
    options.with_validate(options.validate);
    std::move(options).with_validate(options.validate);
  }
  static_assert(std::is_aggregate_v<rlmesh::EnvFactoryServeOptions>);
  {
    rlmesh::EnvFactoryServeOptions options;
    options.with_num_envs(options.num_envs)
        .with_workflow_edition(options.workflow_edition)
        .with_make_kwargs(options.make_kwargs)
        .with_cancel(options.cancel);
    std::move(options)
        .with_num_envs(options.num_envs)
        .with_workflow_edition(options.workflow_edition)
        .with_make_kwargs(options.make_kwargs)
        .with_cancel(options.cancel);
  }
  static_assert(std::is_aggregate_v<rlmesh::ModelOptions>);
  {
    rlmesh::ModelOptions options;
    options.with_spec(options.spec)
        .with_on_episode_end(options.on_episode_end)
        .with_on_close(options.on_close)
        .with_trust_entrypoints(options.trust_entrypoints);
    std::move(options)
        .with_spec(options.spec)
        .with_on_episode_end(options.on_episode_end)
        .with_on_close(options.on_close)
        .with_trust_entrypoints(options.trust_entrypoints);
  }
  static_assert(std::is_aggregate_v<rlmesh::ModelServeOptions>);
  {
    rlmesh::ModelServeOptions options;
    options.with_options(options.options).with_cancel(options.cancel).with_ext(options.ext);
    std::move(options)
        .with_options(options.options)
        .with_cancel(options.cancel)
        .with_ext(options.ext);
  }
  static_assert(std::is_aggregate_v<rlmesh::RemoteEnvOptions>);
  {
    rlmesh::RemoteEnvOptions options;
    options.with_address(options.address)
        .with_host(options.host)
        .with_port(options.port)
        .with_path(options.path)
        .with_transport(options.transport)
        .with_connect_timeout(options.connect_timeout)
        .with_request_timeout(options.request_timeout)
        .with_workflow_edition(options.workflow_edition);
    std::move(options)
        .with_address(options.address)
        .with_host(options.host)
        .with_port(options.port)
        .with_path(options.path)
        .with_transport(options.transport)
        .with_connect_timeout(options.connect_timeout)
        .with_request_timeout(options.request_timeout)
        .with_workflow_edition(options.workflow_edition);
  }
  static_assert(std::is_aggregate_v<rlmesh::View>);
  {
    rlmesh::View options;
    options.with_backend(options.backend)
        .with_port(options.port)
        .with_fps(options.fps)
        .with_source(options.source)
        .with_format(options.format)
        .with_quality(options.quality)
        .with_hold(options.hold)
        .with_step_hz(options.step_hz);
    std::move(options)
        .with_backend(options.backend)
        .with_port(options.port)
        .with_fps(options.fps)
        .with_source(options.source)
        .with_format(options.format)
        .with_quality(options.quality)
        .with_hold(options.hold)
        .with_step_hz(options.step_hz);
  }
  static_assert(std::is_aggregate_v<rlmesh::BindOptions>);
  {
    rlmesh::BindOptions options;
    options.with_close_env(options.close_env)
        .with_execution_horizon(options.execution_horizon)
        .with_workflow_edition(options.workflow_edition);
    std::move(options)
        .with_close_env(options.close_env)
        .with_execution_horizon(options.execution_horizon)
        .with_workflow_edition(options.workflow_edition);
  }
  static_assert(std::is_aggregate_v<rlmesh::RunOptions>);
  {
    rlmesh::RunOptions options;
    options.with_episodes(options.episodes)
        .with_seeds(options.seeds)
        .with_max_episode_steps(options.max_episode_steps)
        .with_max_episode(options.max_episode)
        .with_hooks(options.hooks)
        .with_instruction(options.instruction)
        .with_close_env(options.close_env)
        .with_trust_entrypoints(options.trust_entrypoints)
        .with_execution_horizon(options.execution_horizon)
        .with_prefetch_lead(options.prefetch_lead)
        .with_view(options.view)
        .with_workflow_edition(options.workflow_edition)
        .with_trial_index_base(options.trial_index_base)
        .with_cancel(options.cancel);
    std::move(options)
        .with_episodes(options.episodes)
        .with_seeds(options.seeds)
        .with_max_episode_steps(options.max_episode_steps)
        .with_max_episode(options.max_episode)
        .with_hooks(options.hooks)
        .with_instruction(options.instruction)
        .with_close_env(options.close_env)
        .with_trust_entrypoints(options.trust_entrypoints)
        .with_execution_horizon(options.execution_horizon)
        .with_prefetch_lead(options.prefetch_lead)
        .with_view(options.view)
        .with_workflow_edition(options.workflow_edition)
        .with_trial_index_base(options.trial_index_base)
        .with_cancel(options.cancel);
  }
  static_assert(std::is_aggregate_v<rlmesh::RecorderOptions>);
  {
    rlmesh::RecorderOptions options;
    options.with_result_set_id(options.result_set_id)
        .with_fps(options.fps)
        .with_quality(options.quality);
    std::move(options)
        .with_result_set_id(options.result_set_id)
        .with_fps(options.fps)
        .with_quality(options.quality);
  }
  static_assert(std::is_aggregate_v<rlmesh::RecorderAddOptions>);
  {
    rlmesh::RecorderAddOptions options;
    static_assert(std::is_same_v<decltype(options.config), std::optional<rlmesh::Metadata>>);
    options.with_config(rlmesh::Metadata{}).with_config(std::nullopt);
    options.with_task(options.task)
        .with_config(options.config)
        .with_included_in_metrics(options.included_in_metrics);
    std::move(options)
        .with_task(options.task)
        .with_config(options.config)
        .with_included_in_metrics(options.included_in_metrics);
  }
  static_assert(std::is_aggregate_v<rlmesh::RecorderCaptureOptions>);
  {
    rlmesh::RecorderCaptureOptions options;
    static_assert(std::is_same_v<decltype(options.config), std::optional<rlmesh::Metadata>>);
    options.with_config(rlmesh::Metadata{}).with_config(std::nullopt);
    options.with_task(options.task)
        .with_config(options.config)
        .with_cameras(options.cameras)
        .with_binding(options.binding)
        .with_video_info_keys(options.video_info_keys)
        .with_included_in_metrics(options.included_in_metrics);
    std::move(options)
        .with_task(options.task)
        .with_config(options.config)
        .with_cameras(options.cameras)
        .with_binding(options.binding)
        .with_video_info_keys(options.video_info_keys)
        .with_included_in_metrics(options.included_in_metrics);
  }
  static_assert(std::is_aggregate_v<rlmesh::RecorderToDictOptions>);
  {
    rlmesh::RecorderToDictOptions options;
    options.with_recorded_at(options.recorded_at);
    std::move(options).with_recorded_at(options.recorded_at);
  }
  static_assert(std::is_aggregate_v<rlmesh::RecorderExportOptions>);
  {
    rlmesh::RecorderExportOptions options;
    options.with_archive(options.archive).with_recorded_at(options.recorded_at);
    std::move(options).with_archive(options.archive).with_recorded_at(options.recorded_at);
  }
  static_assert(std::is_aggregate_v<rlmesh::DescribeOptions>);
  {
    rlmesh::DescribeOptions options;
    options.with_kind(options.kind).with_generated_at(options.generated_at);
    std::move(options).with_kind(options.kind).with_generated_at(options.generated_at);
  }
  static_assert(std::is_aggregate_v<rlmesh::DescribeJsonOptions>);
  {
    rlmesh::DescribeJsonOptions options;
    options.with_kind(options.kind)
        .with_generated_at(options.generated_at)
        .with_served_env(options.served_env)
        .with_workflow_edition(options.workflow_edition);
    std::move(options)
        .with_kind(options.kind)
        .with_generated_at(options.generated_at)
        .with_served_env(options.served_env)
        .with_workflow_edition(options.workflow_edition);
  }
}
