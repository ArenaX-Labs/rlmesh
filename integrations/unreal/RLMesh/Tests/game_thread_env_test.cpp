// Standalone test of RLMeshGameThreadEnv.h, the part of the Unreal plugin that
// needs no Unreal headers. The main thread stands in for the game thread: it
// pumps calls while a model drives the served env from another thread.
#include <chrono>
#include <cstdio>
#include <future>
#include <stdexcept>
#include <thread>

#include "RLMeshGameThreadEnv.h"

#define EXPECT(cond)                                                                                                   \
	do                                                                                                                 \
	{                                                                                                                  \
		if (!(cond))                                                                                                   \
		{                                                                                                              \
			std::fprintf(stderr, "%s:%d: expected %s\n", __FILE__, __LINE__, #cond);                                   \
			return 1;                                                                                                  \
		}                                                                                                              \
	} while (0)

int main()
try
{
	const std::thread::id game_thread = std::this_thread::get_id();
	auto thread = std::make_shared<FRLMeshGameThread>();
	int64_t position = 0;
	bool every_call_on_game_thread = true;

	FRLMeshGameThreadEnv env(
		thread, rlmesh::spaces::Discrete(10), rlmesh::spaces::Discrete(2), "Counter-v0",
		[&](std::optional<int64_t>, const std::optional<rlmesh::InfoDict>&)
		{
			every_call_on_game_thread &= std::this_thread::get_id() == game_thread;
			position = 0;
			return rlmesh::ResetResult{rlmesh::Value(position), {}};
		},
		[&](std::optional<rlmesh::Value>)
		{
			every_call_on_game_thread &= std::this_thread::get_id() == game_thread;
			++position;
			return rlmesh::StepResult{rlmesh::Value(position), 1.0, position >= 3, false, {}};
		});

	rlmesh::EnvServer server(env, rlmesh::EnvServerOptions{}.with_address(":0/env"));
	server.start();
	const std::string address = server.address();

	// A model on another thread; this thread pumps like a game tick until it ends.
	auto run = std::async(std::launch::async,
						  [&]
						  {
							  rlmesh::Model model(
								  [](const rlmesh::Value&)
								  {
									  return rlmesh::Value(1);
								  });
							  return model.run(address, rlmesh::RunOptions{}.with_episodes(2));
						  });
	while (run.wait_for(std::chrono::milliseconds(1)) != std::future_status::ready)
		thread->Pump();
	rlmesh::RunResult result = run.get();
	EXPECT(result.num_episodes() == 2);
	EXPECT(result.total_steps() == 6);
	EXPECT(every_call_on_game_thread);

	// Close() fails a call that is waiting for a tick, and every later call.
	auto blocked = std::async(std::launch::async,
							  [&]
							  {
								  return thread->Call(
									  []
									  {
										  return 1;
									  });
							  });
	std::this_thread::sleep_for(std::chrono::milliseconds(50));
	thread->Close();
	bool failed = false;
	try
	{
		static_cast<void>(blocked.get());
	}
	catch (const std::runtime_error&)
	{
		failed = true;
	}
	EXPECT(failed);
	failed = false;
	try
	{
		static_cast<void>(thread->Call(
			[]
			{
				return 1;
			}));
	}
	catch (const std::runtime_error&)
	{
		failed = true;
	}
	EXPECT(failed);

	server.shutdown();
	std::printf("game thread env OK\n");
	return 0;
}
catch (const std::exception& error)
{
	std::fprintf(stderr, "error: %s\n", error.what());
	return 1;
}
