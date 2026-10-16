// Serves an RLMesh env whose reset and step run on Unreal's game thread.
//
// An EnvServer calls reset and step on its own threads, but Unreal game state
// may only be touched on the game thread. FRLMeshGameThreadEnv posts each call
// to an FRLMeshGameThread and blocks until the game thread runs it from its
// tick. This header needs only rlmesh.hpp and the standard library, so it also
// builds outside Unreal.
#pragma once

#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include "RLMeshIncludes.h"

// Calls posted from server threads, run by whichever thread calls Pump().
class FRLMeshGameThread
{
public:
	// On a server thread: runs Fn on the pumping thread and returns its result,
	// rethrowing what it throws. Throws once Close() has been called.
	template <class Fn> auto Call(Fn&& Function) -> decltype(Function())
	{
		using Result = decltype(Function());
		auto Promise = std::make_shared<std::promise<Result>>();
		std::future<Result> Future = Promise->get_future();
		FJob Job;
		Job.Run = [Promise, Function = std::forward<Fn>(Function)]() mutable
		{
			try
			{
				Promise->set_value(Function());
			}
			catch (...)
			{
				Promise->set_exception(std::current_exception());
			}
		};
		Job.Fail = [Promise](std::exception_ptr Error)
		{
			Promise->set_exception(Error);
		};
		{
			std::lock_guard<std::mutex> Lock(Mutex);
			if (bClosed)
				throw std::runtime_error(ClosedMessage);
			Queue.push_back(std::move(Job));
		}
		return Future.get();
	}

	// On the game thread, once per tick: runs every call posted so far.
	void Pump()
	{
		std::deque<FJob> Jobs;
		{
			std::lock_guard<std::mutex> Lock(Mutex);
			Jobs.swap(Queue);
		}
		for (FJob& Job : Jobs)
			Job.Run();
	}

	// Fails pending and later calls, so a server thread blocked in Call() returns
	// and the server can shut down without waiting for another tick.
	void Close()
	{
		std::deque<FJob> Jobs;
		{
			std::lock_guard<std::mutex> Lock(Mutex);
			bClosed = true;
			Jobs.swap(Queue);
		}
		std::exception_ptr Error = std::make_exception_ptr(std::runtime_error(ClosedMessage));
		for (FJob& Job : Jobs)
			Job.Fail(Error);
	}

private:
	struct FJob
	{
		std::function<void()> Run;
		std::function<void(std::exception_ptr)> Fail;
	};
	static constexpr const char* ClosedMessage = "the Unreal world serving this env has ended";

	std::mutex Mutex;
	std::deque<FJob> Queue;
	bool bClosed = false;
};

// An env for rlmesh::EnvServer whose reset and step run through a game thread.
class FRLMeshGameThreadEnv
{
public:
	using FReset =
		std::function<rlmesh::ResetResult(std::optional<int64_t> Seed, const std::optional<rlmesh::InfoDict>& Options)>;
	using FStep = std::function<rlmesh::StepResult(std::optional<rlmesh::Value> Action)>;

	FRLMeshGameThreadEnv(std::shared_ptr<FRLMeshGameThread> InGameThread, rlmesh::spaces::Space InObservationSpace,
						 rlmesh::spaces::Space InActionSpace, std::string InId, FReset InReset, FStep InStep)
		: GameThread(std::move(InGameThread)), ObservationSpace(std::move(InObservationSpace)),
		  ActionSpace(std::move(InActionSpace)), Id(std::move(InId)), Reset(std::move(InReset)), Step(std::move(InStep))
	{
	}

	rlmesh::spaces::Space observation_space() const { return ObservationSpace; }
	rlmesh::spaces::Space action_space() const { return ActionSpace; }
	std::optional<std::string> id() const { return Id; }

	rlmesh::ResetResult reset(std::optional<int64_t> Seed, const std::optional<rlmesh::InfoDict>& Options)
	{
		return GameThread->Call(
			[&]
			{
				return Reset(Seed, Options);
			});
	}

	rlmesh::StepResult step(std::optional<rlmesh::Value> Action)
	{
		return GameThread->Call(
			[&]
			{
				return Step(std::move(Action));
			});
	}

private:
	std::shared_ptr<FRLMeshGameThread> GameThread;
	rlmesh::spaces::Space ObservationSpace;
	rlmesh::spaces::Space ActionSpace;
	std::string Id;
	FReset Reset;
	FStep Step;
};
