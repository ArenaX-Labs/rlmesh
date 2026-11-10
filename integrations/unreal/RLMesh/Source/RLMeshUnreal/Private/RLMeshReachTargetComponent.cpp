#include "RLMeshReachTargetComponent.h"

#include "Components/SceneComponent.h"
#include "GameFramework/Actor.h"

#include <cstddef>
#include <limits>

namespace
{
FVector ActionAsDirection(const rlmesh::Value& Action)
{
	if (Action.kind() == rlmesh::ValueKind::Int)
	{
		// Backward-compatible Discrete(3): 0=-X, 1=stay, 2=+X.
		const int64_t Move = Action.as_int() - 1;
		return FVector(static_cast<double>(Move), 0.0, 0.0);
	}

	const rlmesh::Tensor Tensor = Action.as_tensor();
	if (Tensor.dtype() != rlmesh::DType::Float32 || Tensor.size() < 3)
	{
		throw rlmesh::RecoverableEnvironmentException(
			"step needs a float32 action of shape (3,) in [-1, 1], or Discrete 0/1/2");
	}
	const rlmesh::Span<const std::byte> Bytes = Tensor.buffer();
	if (Bytes.size() < 3 * sizeof(float))
	{
		throw rlmesh::RecoverableEnvironmentException("step action tensor buffer is too small");
	}
	const float* Data = reinterpret_cast<const float*>(Bytes.data());
	return FVector(static_cast<double>(Data[0]), static_cast<double>(Data[1]),
				   static_cast<double>(Data[2]));
}
} // namespace

void URLMeshReachTargetComponent::BeginPlay()
{
	AActor* Owner = GetOwner();
	if (Owner)
	{
		if (USceneComponent* Root = Owner->GetRootComponent())
		{
			if (Root->Mobility == EComponentMobility::Static)
			{
				Root->SetMobility(EComponentMobility::Movable);
				UE_LOG(LogRLMesh, Log,
					   TEXT("Reach Target: set %s root mobility to Movable so steps can move it"),
					   *Owner->GetName());
			}
		}
		// Remember the start before Super::BeginPlay starts serving.
		Start = Owner->GetActorLocation();
	}
	Random.GenerateNewSeed();
	Super::BeginPlay();
}

rlmesh::spaces::Space URLMeshReachTargetComponent::ObservationSpace() const
{
	const double Infinity = std::numeric_limits<double>::infinity();
	return rlmesh::spaces::Box(-Infinity, Infinity, {3});
}

rlmesh::spaces::Space URLMeshReachTargetComponent::ActionSpace() const
{
	return rlmesh::spaces::Box(-1.0, 1.0, {3});
}

rlmesh::ResetResult URLMeshReachTargetComponent::ResetEnv(std::optional<int64_t> Seed,
														  const std::optional<rlmesh::InfoDict>& /*Options*/)
{
	if (Seed)
	{
		// FRandomStream takes int32; fold both halves of the int64 seed so high
		// bits are not discarded by a narrowing cast.
		const uint64 Bits = static_cast<uint64>(*Seed);
		Random.Initialize(static_cast<int32>(Bits ^ (Bits >> 32)));
	}
	AActor* Owner = GetOwner();
	Owner->SetActorLocation(Start);
	if (IsValid(TargetActor) && TargetActor != Owner)
	{
		TargetOffset = TargetActor->GetActorLocation() - Start;
	}
	else
	{
		TargetOffset = FVector(Random.FRandRange(-TargetRange, TargetRange),
							   Random.FRandRange(-TargetRange, TargetRange),
							   Random.FRandRange(-TargetRange, TargetRange));
	}
	Steps = 0;
	const FVector Loc = Owner->GetActorLocation();
	const FVector Goal = Start + TargetOffset;
	UE_LOG(LogRLMesh, Log,
		   TEXT("Reach Target reset: %s at (%.1f, %.1f, %.1f) goal (%.1f, %.1f, %.1f) dist=%.1f"),
		   *Owner->GetName(), Loc.X, Loc.Y, Loc.Z, Goal.X, Goal.Y, Goal.Z, Distance());
	return {Observe(), MakeInfo()};
}

rlmesh::StepResult URLMeshReachTargetComponent::StepEnv(const std::optional<rlmesh::Value>& Action)
{
	if (!Action)
	{
		throw rlmesh::RecoverableEnvironmentException(
			"step needs a float32 action of shape (3,) in [-1, 1]");
	}

	FVector Dir = ActionAsDirection(*Action);
	AActor* Owner = GetOwner();
	if (!Dir.IsNearlyZero())
	{
		Dir.Normalize();
		Owner->AddActorWorldOffset(Dir * StepSize);
	}
	++Steps;

	const double Dist = Distance();
	const bool bReached = Dist <= StepSize / 2.0;
	// Keep reward in roughly [-1, 0] even when TargetActor is farther than TargetRange.
	const double RewardScale =
		FMath::Max(static_cast<double>(TargetRange), TargetOffset.Size());
	const FVector Loc = Owner->GetActorLocation();
	const FVector Rem = Remaining();
	UE_LOG(LogRLMesh, Log,
		   TEXT("Reach Target step %d: rem=(%.1f, %.1f, %.1f) -> (%.1f, %.1f, %.1f) dist=%.1f%s%s"),
		   Steps, Rem.X, Rem.Y, Rem.Z, Loc.X, Loc.Y, Loc.Z, Dist,
		   bReached ? TEXT(" reached") : TEXT(""),
		   (!bReached && Steps >= MaxSteps) ? TEXT(" truncated") : TEXT(""));
	return {Observe(), -Dist / RewardScale, bReached, !bReached && Steps >= MaxSteps, MakeInfo()};
}

FVector URLMeshReachTargetComponent::Remaining() const
{
	const FVector Goal = Start + TargetOffset;
	return Goal - GetOwner()->GetActorLocation();
}

double URLMeshReachTargetComponent::Distance() const
{
	return Remaining().Size();
}

rlmesh::Value URLMeshReachTargetComponent::Observe() const
{
	const FVector Rem = Remaining();
	const float Observation[3] = {static_cast<float>(Rem.X), static_cast<float>(Rem.Y),
								  static_cast<float>(Rem.Z)};
	return rlmesh::Tensor(Observation, sizeof(Observation), {3}, rlmesh::DType::Float32);
}

rlmesh::InfoDict URLMeshReachTargetComponent::MakeInfo() const
{
	const FVector Loc = GetOwner()->GetActorLocation();
	const FVector Goal = Start + TargetOffset;
	const float AgentXyz[3] = {static_cast<float>(Loc.X), static_cast<float>(Loc.Y),
							   static_cast<float>(Loc.Z)};
	const float TargetXyz[3] = {static_cast<float>(Goal.X), static_cast<float>(Goal.Y),
								static_cast<float>(Goal.Z)};
	rlmesh::InfoDict Info;
	Info.insert_or_assign(
		"agent_xyz", rlmesh::Tensor(AgentXyz, sizeof(AgentXyz), {3}, rlmesh::DType::Float32));
	Info.insert_or_assign(
		"target_xyz", rlmesh::Tensor(TargetXyz, sizeof(TargetXyz), {3}, rlmesh::DType::Float32));
	Info.insert_or_assign("distance", Distance());
	Info.insert_or_assign("steps", static_cast<int64_t>(Steps));
	return Info;
}
