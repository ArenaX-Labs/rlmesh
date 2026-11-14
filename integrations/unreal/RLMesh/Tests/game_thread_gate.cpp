// The Unreal plugin's gate program of python/rlmesh/tests/conformance/native_gate.py:
// the C++ echo env served as URLMeshEnvComponent serves a world, every call run
// by a game thread that pumps like a tick. The plugin serves envs only.
#include <atomic>
#include <chrono>
#include <thread>

#include "RLMeshGameThreadEnv.h"
#include "cpp_gate.hpp"

int main(int argc, char** argv)
try
{
	const std::string verb = argc > 1 ? argv[1] : "";
	if (verb != "env" || argc != 4)
		throw gate::Unsupported{"the Unreal plugin serves envs only"};
	const std::string row = argv[2];
	if (row == "render")
		throw gate::Unsupported{"the Unreal plugin's env renders no frame"};
	auto env = std::make_shared<gate::EchoEnv>(row);
	auto thread = std::make_shared<FRLMeshGameThread>();
	FRLMeshGameThreadEnv served(
		thread, env->observation_space(), env->action_space(), "Gate-v0",
		[env](std::optional<int64_t> Seed, const std::optional<rlmesh::InfoDict>& Options)
		{
			return env->reset(Seed, Options);
		},
		[env](std::optional<rlmesh::Value> Action)
		{
			return env->step(std::move(Action));
		});
	rlmesh::EnvServer server(served, rlmesh::EnvServerOptions{}.with_address(argv[3]));
	server.start();
	std::printf("listening on %s\n", server.address().c_str());
	std::fflush(stdout);
	std::atomic<bool> ended{false};
	std::thread(
		[&ended]
		{
			gate::wait_for_eof();
			ended = true;
		})
		.detach();
	while (!ended)
	{
		thread->Pump();
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
	thread->Close();
	server.shutdown();
	return 0;
}
catch (const gate::Unsupported& unsupported)
{
	std::printf("unsupported: %s\n", unsupported.why.c_str());
	return 0;
}
catch (const std::exception& error)
{
	std::fprintf(stderr, "%s\n", error.what());
	return 1;
}
