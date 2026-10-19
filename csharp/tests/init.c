/* Field-wise INIT checks deliberately ignore implicit padding. */
#include "rlmesh.h"
#include <stdio.h>

#define NUM(name, field) printf("\"" #field "\":%lld,", (long long)name.field)
#define LAST(name, field) printf("\"" #field "\":%lld}", (long long)name.field)
#define PTR(name, field) printf("\"" #field "\":%s,", name.field ? "true" : "false")
#define LASTPTR(name, field) printf("\"" #field "\":%s}", name.field ? "true" : "false")
#define ZERO(name, field) do { int zero = 1; for (size_t i = 0; i < sizeof(name.field); i++) if (((const unsigned char*)name.field)[i]) zero = 0; printf("\"" #field "\":%s,", zero ? "true" : "false"); } while (0)

int main(void) {
  RLMeshBuildInfo build = RLMESH_BUILD_INFO_INIT;
  RLMeshBoxOptions box = RLMESH_BOX_OPTIONS_INIT;
  RLMeshDiscreteOptions discrete = RLMESH_DISCRETE_OPTIONS_INIT;
  RLMeshServeOptions serve = RLMESH_SERVE_OPTIONS_INIT;
  RLMeshEnvServerOptions server = RLMESH_ENV_SERVER_OPTIONS_INIT;
  RLMeshExtEnvServerOptions ext = RLMESH_EXT_ENV_SERVER_OPTIONS_INIT;
  RLMeshEnvFactoryServeOptions factory = RLMESH_ENV_FACTORY_SERVE_OPTIONS_INIT;
  RLMeshExtEnvFactoryServeOptions factory_ext = RLMESH_EXT_ENV_FACTORY_SERVE_OPTIONS_INIT;
  RLMeshExtModelServeOptions model_ext = RLMESH_EXT_MODEL_SERVE_OPTIONS_INIT;
  RLMeshEnvAttributes attributes = RLMESH_ENV_ATTRIBUTES_INIT;
  RLMeshEnvVtable vtable = RLMESH_ENV_VTABLE_INIT;
  printf("{\"version\":\"%s\",\"build\":{", RLMESH_VERSION);
  NUM(build, struct_size); PTR(build, version); PTR(build, protocol_generation);
  PTR(build, workflow_edition); PTR(build, workflow_edition_base); PTR(build, supported_workflow_editions);
  NUM(build, supported_workflow_editions_len); PTR(build, build_cohort); NUM(build, build_source); NUM(build, _reserved); LASTPTR(build, git);
  printf(",\"box\":{"); NUM(box, struct_size); NUM(box, dtype); LAST(box, _reserved);
  printf(",\"discrete\":{"); NUM(discrete, struct_size); NUM(discrete, start); NUM(discrete, dtype); LAST(discrete, _reserved);
  printf(",\"serve\":{"); NUM(serve, struct_size); NUM(serve, allow_remote_shutdown); NUM(serve, has_idle_timeout_seconds); NUM(serve, has_drain_timeout_seconds); NUM(serve, has_close_timeout_seconds); NUM(serve, compress_responses); ZERO(serve, _reserved); NUM(serve, idle_timeout_seconds); NUM(serve, drain_timeout_seconds); NUM(serve, close_timeout_seconds); LASTPTR(serve, workflow_edition);
  printf(",\"server\":{"); NUM(server, struct_size); PTR(server, address); PTR(server, host); PTR(server, path); PTR(server, options); PTR(server, tags); PTR(server, ext); NUM(server, port); NUM(server, has_port); NUM(server, has_transport); NUM(server, transport); ZERO(server, _reserved); LAST(server, close_env_on_shutdown);
  printf(",\"ext\":{"); NUM(ext, struct_size); PTR(ext, token); PTR(ext, peer_info_extra); PTR(ext, host_runtime_json); ZERO(ext, _reserved); LAST(ext, publish_describe);
  printf(",\"factory\":{"); NUM(factory, struct_size); NUM(factory, num_envs); PTR(factory, workflow_edition); PTR(factory, make_kwargs); PTR(factory, cancel); NUM(factory, _reserved); LASTPTR(factory, ext);
  printf(",\"factory_ext\":{"); NUM(factory_ext, struct_size); LASTPTR(factory_ext, host_runtime_json);
  printf(",\"model_ext\":{"); NUM(model_ext, struct_size); PTR(model_ext, token); NUM(model_ext, predict_concurrency); NUM(model_ext, has_predict_concurrency); ZERO(model_ext, _reserved); ZERO(model_ext, _reserved1); LAST(model_ext, no_signal_handlers);
  printf(",\"attributes\":{"); NUM(attributes, struct_size); PTR(attributes, observation_space); PTR(attributes, action_space); PTR(attributes, render_mode); PTR(attributes, metadata); LASTPTR(attributes, id);
  printf(",\"vtable\":{"); NUM(vtable, struct_size); PTR(vtable, reset); PTR(vtable, step); PTR(vtable, render); PTR(vtable, close); LASTPTR(vtable, free_user_data);
  printf("}\n");
  return 0;
}
