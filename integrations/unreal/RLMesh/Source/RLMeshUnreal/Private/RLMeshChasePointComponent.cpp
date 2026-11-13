#include "RLMeshChasePointComponent.h"

#include "Components/SceneComponent.h"
#include "GameFramework/Actor.h"

#include <cstddef>
#include <limits>

namespace ChasePointDetail
{
FVector2D ActionAsDirectionXY(const rlmesh::Value& Action)
{
	const rlmesh::Tensor Tensor = Action.as_tensor();
	if (Tensor.dtype() != rlmesh::DType::Float32 || Tensor.size() < 2)
	{
		throw rlmesh::RecoverableEnvironmentException(
			"step needs a float32 action of shape (2,) in [-1, 1]");
	}
	const rlmesh::Span<const std::byte> Bytes = Tensor.buffer();
	if (Bytes.size() < 2 * sizeof(float))
	{
		throw rlmesh::RecoverableEnvironmentException("step action tensor buffer is too small");
	}
	const float* Data = reinterpret_cast<const float*>(Bytes.data());
	return FVector2D(static_cast<double>(Data[0]), static_cast<double>(Data[1]));
}
} // namespace ChasePointDetail

URLMeshChasePointComponent::URLMeshChasePointComponent()
{
	EnvId = TEXT("Unreal-ChasePoint-v0");
}

void URLMeshChasePointComponent::BeginPlay()
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
					   TEXT("Chase Point: set %s root mobility to Movable so steps can move it"),
					   *Owner->GetName());
			}
		}
		Start = Owner->GetActorLocation();
	}
	if (IsValid(GoalActor))
	{
		if (USceneComponent* GoalRoot = GoalActor->GetRootComponent())
		{
			if (GoalRoot->Mobility == EComponentMobility::Static)
			{
				GoalRoot->SetMobility(EComponentMobility::Movable);
			}
		}
	}
	Random.GenerateNewSeed();
	Super::BeginPlay();
}

rlmesh::spaces::Space URLMeshChasePointComponent::ObservationSpace() const
{
	const double Infinity = std::numeric_limits<double>::infinity();
	return rlmesh::spaces::Box(-Infinity, Infinity, {2});
}

rlmesh::spaces::Space URLMeshChasePointComponent::ActionSpace() const
{
	return rlmesh::spaces::Box(-1.0, 1.0, {2});
}

void URLMeshChasePointComponent::LockOwnerZ() const
{
	AActor* Owner = GetOwner();
	if (!Owner)
	{
		return;
	}
	FVector Loc = Owner->GetActorLocation();
	if (!FMath::IsNearlyEqual(Loc.Z, Start.Z))
	{
		Loc.Z = Start.Z;
		Owner->SetActorLocation(Loc, false, nullptr, ETeleportType::TeleportPhysics);
	}
}

void URLMeshChasePointComponent::SyncGoalActor() const
{
	if (!IsValid(GoalActor) || GoalActor == GetOwner())
	{
		return;
	}
	const FVector GoalWorld(Start.X + GoalOffset.X, Start.Y + GoalOffset.Y, Start.Z);
	GoalActor->SetActorLocation(GoalWorld, false, nullptr, ETeleportType::TeleportPhysics);
}

void URLMeshChasePointComponent::WanderGoal()
{
	if (GoalStepSize <= 0.f || GoalMoveSteps <= 0)
	{
		return;
	}

	const FVector2D GoalXY(Start.X + GoalOffset.X, Start.Y + GoalOffset.Y);
	const FVector AgentLoc = GetOwner()->GetActorLocation();
	const FVector2D TowardAgent(AgentLoc.X - GoalXY.X, AgentLoc.Y - GoalXY.Y);

	// Random heading on the floor; flip if it would move toward the agent.
	const float Angle = Random.FRandRange(0.f, 2.f * PI);
	FVector2D Dir(FMath::Cos(Angle), FMath::Sin(Angle));
	if (!TowardAgent.IsNearlyZero() && FVector2D::DotProduct(Dir, TowardAgent) > 0.0)
	{
		Dir *= -1.0;
	}

	FVector2D Next = GoalXY + Dir * GoalStepSize;
	FVector2D FromHome = Next - GoalHome;
	const double HomeDist = FromHome.Size();
	if (HomeDist > GoalWanderRadius && HomeDist > 1e-6)
	{
		FromHome *= (GoalWanderRadius / HomeDist);
		Next = GoalHome + FromHome;
	}

	GoalOffset = FVector2D(Next.X - Start.X, Next.Y - Start.Y);
	SyncGoalActor();
}

rlmesh::ResetResult URLMeshChasePointComponent::ResetEnv(
	std::optional<int64_t> Seed, const std::optional<rlmesh::InfoDict>& /*Options*/)
{
	if (Seed)
	{
		const uint64 Bits = static_cast<uint64>(*Seed);
		Random.Initialize(static_cast<int32>(Bits ^ (Bits >> 32)));
	}
	AActor* Owner = GetOwner();
	Owner->SetActorLocation(Start, false, nullptr, ETeleportType::TeleportPhysics);
	if (IsValid(GoalActor) && GoalActor != Owner)
	{
		if (!bHaveGoalSpawn)
		{
			const FVector GoalLoc = GoalActor->GetActorLocation();
			GoalSpawnOffset = FVector2D(GoalLoc.X - Start.X, GoalLoc.Y - Start.Y);
			bHaveGoalSpawn = true;
		}
		GoalOffset = GoalSpawnOffset;
	}
	else
	{
		GoalOffset = FVector2D(Random.FRandRange(-GoalRange, GoalRange),
							   Random.FRandRange(-GoalRange, GoalRange));
	}
	GoalHome = FVector2D(Start.X + GoalOffset.X, Start.Y + GoalOffset.Y);
	Steps = 0;
	SyncGoalActor();
	const FVector Loc = Owner->GetActorLocation();
	UE_LOG(LogRLMesh, Log,
		   TEXT("Chase Point reset: %s at (%.1f, %.1f, %.1f) goal XY (%.1f, %.1f) dist=%.1f "
				"(wander %d steps @ %.1f within r=%.1f)"),
		   *Owner->GetName(), Loc.X, Loc.Y, Loc.Z, GoalHome.X, GoalHome.Y, DistanceXY(), GoalMoveSteps,
		   GoalStepSize, GoalWanderRadius);
	return {Observe(), MakeInfo()};
}

rlmesh::StepResult URLMeshChasePointComponent::StepEnv(const std::optional<rlmesh::Value>& Action)
{
	if (!Action)
	{
		throw rlmesh::RecoverableEnvironmentException(
			"step needs a float32 action of shape (2,) in [-1, 1]");
	}

	FVector2D Dir = ChasePointDetail::ActionAsDirectionXY(*Action);
	AActor* Owner = GetOwner();
	const double DistBefore = DistanceXY();
	if (!Dir.IsNearlyZero() && DistBefore > 0.0)
	{
		Dir.Normalize();
		const double Travel = FMath::Min(static_cast<double>(StepSize), DistBefore);
		const FVector Loc = Owner->GetActorLocation();
		Owner->SetActorLocation(
			FVector(Loc.X + Dir.X * Travel, Loc.Y + Dir.Y * Travel, Start.Z), false, nullptr,
			ETeleportType::TeleportPhysics);
	}
	else
	{
		LockOwnerZ();
	}

	++Steps;
	const bool bGoalMoving = Steps <= GoalMoveSteps;
	if (bGoalMoving)
	{
		WanderGoal();
	}

	const double Dist = DistanceXY();
	const bool bReached = Dist <= FMath::Max(1.0, static_cast<double>(StepSize) * 0.05);
	const double HomeOffset =
		FVector2D(GoalHome.X - Start.X, GoalHome.Y - Start.Y).Size();
	const double RewardScale = FMath::Max(static_cast<double>(GoalRange), HomeOffset);
	const FVector Loc = Owner->GetActorLocation();
	const FVector2D Rem = RemainingXY();
	UE_LOG(LogRLMesh, Log,
		   TEXT("Chase Point step %d: rem=(%.1f, %.1f) -> (%.1f, %.1f, %.1f) dist=%.1f%s%s%s"),
		   Steps, Rem.X, Rem.Y, Loc.X, Loc.Y, Loc.Z, Dist, bGoalMoving ? TEXT(" goal_moving") : TEXT(""),
		   bReached ? TEXT(" reached") : TEXT(""),
		   (!bReached && Steps >= MaxSteps) ? TEXT(" truncated") : TEXT(""));
	return {Observe(), -Dist / RewardScale, bReached, !bReached && Steps >= MaxSteps, MakeInfo()};
}

FVector2D URLMeshChasePointComponent::RemainingXY() const
{
	const FVector Loc = GetOwner()->GetActorLocation();
	return FVector2D(Start.X + GoalOffset.X - Loc.X, Start.Y + GoalOffset.Y - Loc.Y);
}

double URLMeshChasePointComponent::DistanceXY() const
{
	return RemainingXY().Size();
}

rlmesh::Value URLMeshChasePointComponent::Observe() const
{
	const FVector2D Rem = RemainingXY();
	const float Observation[2] = {static_cast<float>(Rem.X), static_cast<float>(Rem.Y)};
	return rlmesh::Tensor(Observation, sizeof(Observation), {2}, rlmesh::DType::Float32);
}

rlmesh::InfoDict URLMeshChasePointComponent::MakeInfo() const
{
	const FVector Loc = GetOwner()->GetActorLocation();
	const float AgentXy[2] = {static_cast<float>(Loc.X), static_cast<float>(Loc.Y)};
	const float GoalXy[2] = {static_cast<float>(Start.X + GoalOffset.X),
							 static_cast<float>(Start.Y + GoalOffset.Y)};
	rlmesh::InfoDict Info;
	Info.insert_or_assign(
		"agent_xy", rlmesh::Tensor(AgentXy, sizeof(AgentXy), {2}, rlmesh::DType::Float32));
	Info.insert_or_assign(
		"goal_xy", rlmesh::Tensor(GoalXy, sizeof(GoalXy), {2}, rlmesh::DType::Float32));
	Info.insert_or_assign("distance", DistanceXY());
	Info.insert_or_assign("steps", static_cast<int64_t>(Steps));
	Info.insert_or_assign("goal_moving", static_cast<int64_t>(Steps <= GoalMoveSteps ? 1 : 0));
	return Info;
}
