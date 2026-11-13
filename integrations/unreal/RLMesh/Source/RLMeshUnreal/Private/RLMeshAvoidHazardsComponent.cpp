#include "RLMeshAvoidHazardsComponent.h"

#include "Components/SceneComponent.h"
#include "Components/StaticMeshComponent.h"
#include "Engine/StaticMesh.h"
#include "Engine/StaticMeshActor.h"
#include "Engine/World.h"
#include "GameFramework/Actor.h"
#include "UObject/UObjectGlobals.h"

#include <cstddef>
#include <limits>
#include <vector>

namespace AvoidHazardsDetail
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

bool AabbOverlapXY(const FVector2D& A, float AHalf, const FVector2D& B, float BHalf)
{
	return FMath::Abs(A.X - B.X) <= (AHalf + BHalf) && FMath::Abs(A.Y - B.Y) <= (AHalf + BHalf);
}
} // namespace AvoidHazardsDetail

URLMeshAvoidHazardsComponent::URLMeshAvoidHazardsComponent()
{
	EnvId = TEXT("Unreal-AvoidHazards-v0");
}

void URLMeshAvoidHazardsComponent::BeginPlay()
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
					   TEXT("Avoid Hazards: set %s root mobility to Movable so steps can move it"),
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
	EnsureHazardActors();
	Super::BeginPlay();
}

void URLMeshAvoidHazardsComponent::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	DestroyHazardActors();
	Super::EndPlay(EndPlayReason);
}

rlmesh::spaces::Space URLMeshAvoidHazardsComponent::ObservationSpace() const
{
	const double Infinity = std::numeric_limits<double>::infinity();
	return rlmesh::spaces::Box(-Infinity, Infinity, {ObsDim()});
}

rlmesh::spaces::Space URLMeshAvoidHazardsComponent::ActionSpace() const
{
	return rlmesh::spaces::Box(-1.0, 1.0, {2});
}

void URLMeshAvoidHazardsComponent::LockOwnerZ() const
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

void URLMeshAvoidHazardsComponent::SyncGoalActor() const
{
	if (!IsValid(GoalActor) || GoalActor == GetOwner())
	{
		return;
	}
	const FVector GoalWorld(Start.X + GoalOffset.X, Start.Y + GoalOffset.Y, Start.Z);
	GoalActor->SetActorLocation(GoalWorld, false, nullptr, ETeleportType::TeleportPhysics);
}

void URLMeshAvoidHazardsComponent::EnsureHazardActors()
{
	UWorld* World = GetWorld();
	if (!World)
	{
		return;
	}
	const int32 N = FMath::Max(1, NumHazards);
	HazardOffsets.SetNum(N);
	while (HazardActors.Num() > N)
	{
		AActor* Extra = HazardActors.Pop();
		if (IsValid(Extra))
		{
			Extra->Destroy();
		}
	}
	UStaticMesh* Mesh =
		LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Cube.Cube"));
	const float Scale = (HazardHalfExtent * 2.f) / 100.f; // BasicShapes Cube is 100uu
	while (HazardActors.Num() < N)
	{
		FActorSpawnParameters Params;
		Params.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
		AStaticMeshActor* Hazard =
			World->SpawnActor<AStaticMeshActor>(Start, FRotator::ZeroRotator, Params);
		if (!Hazard)
		{
			break;
		}
		Hazard->SetActorLabel(FString::Printf(TEXT("Hazard_%d"), HazardActors.Num()));
		if (UStaticMeshComponent* SMC = Hazard->GetStaticMeshComponent())
		{
			SMC->SetMobility(EComponentMobility::Movable);
			if (Mesh)
			{
				SMC->SetStaticMesh(Mesh);
			}
			SMC->SetWorldScale3D(FVector(Scale));
			SMC->SetCollisionEnabled(ECollisionEnabled::NoCollision);
			if (HazardMaterial)
			{
				SMC->SetMaterial(0, HazardMaterial);
			}
		}
		HazardActors.Add(Hazard);
	}
}

void URLMeshAvoidHazardsComponent::DestroyHazardActors()
{
	for (AActor* Hazard : HazardActors)
	{
		if (IsValid(Hazard))
		{
			Hazard->Destroy();
		}
	}
	HazardActors.Reset();
}

void URLMeshAvoidHazardsComponent::SyncHazardActors() const
{
	const int32 N = FMath::Min(HazardActors.Num(), HazardOffsets.Num());
	for (int32 i = 0; i < N; ++i)
	{
		AActor* Hazard = HazardActors[i];
		if (!IsValid(Hazard))
		{
			continue;
		}
		const FVector2D& Off = HazardOffsets[i];
		Hazard->SetActorLocation(FVector(Start.X + Off.X, Start.Y + Off.Y, Start.Z), false, nullptr,
								 ETeleportType::TeleportPhysics);
		Hazard->SetActorHiddenInGame(false);
	}
}

bool URLMeshAvoidHazardsComponent::SampleLayout()
{
	const int32 N = FMath::Max(1, NumHazards);
	HazardOffsets.SetNum(N);

	auto FarEnough = [this](const FVector2D& A, const FVector2D& B) -> bool
	{
		return (A - B).Size() >= MinClearance;
	};

	auto ChooseGoal = [this, &FarEnough]() -> bool
	{
		if (IsValid(GoalActor) && GoalActor != GetOwner())
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
		return FarEnough(FVector2D::ZeroVector, GoalOffset);
	};

	// Place hazards along the start→goal corridor so a straight run is blocked.
	// Overlap on the centerline requires |lateral| <= AgentHalfExtent + HazardHalfExtent.
	const float BlockHalf = AgentHalfExtent + HazardHalfExtent;
	auto PlaceOnCorridor = [this, N, BlockHalf, &FarEnough]() -> bool
	{
		const double PathLen = GoalOffset.Size();
		if (PathLen < MinClearance * 2.0)
		{
			return false;
		}
		const FVector2D PathDir = GoalOffset / PathLen;
		const FVector2D Perp(-PathDir.Y, PathDir.X);
		const float EndPad = FMath::Clamp(MinClearance / static_cast<float>(PathLen), 0.15f, 0.35f);

		for (int32 i = 0; i < N; ++i)
		{
			bool bPlaced = false;
			for (int32 Inner = 0; Inner < 48; ++Inner)
			{
				const float T = EndPad + (1.f - 2.f * EndPad) * (static_cast<float>(i) + Random.FRand()) /
											   static_cast<float>(N);
				float Lateral;
				if (i == 0)
				{
					// First hazard sits on the centerline so the direct path is contested.
					Lateral = Random.FRandRange(-HazardHalfExtent * 0.25f, HazardHalfExtent * 0.25f);
				}
				else if (Inner < 16)
				{
					Lateral = Random.FRandRange(-BlockHalf * 0.75f, BlockHalf * 0.75f);
				}
				else
				{
					Lateral = Random.FRandRange(-CorridorHalfWidth, CorridorHalfWidth);
				}
				const FVector2D Cand = GoalOffset * T + Perp * Lateral;
				if (!FarEnough(Cand, FVector2D::ZeroVector) || !FarEnough(Cand, GoalOffset))
				{
					continue;
				}
				bool bClear = true;
				for (int32 j = 0; j < i; ++j)
				{
					if ((Cand - HazardOffsets[j]).Size() < (HazardHalfExtent * 2.f + 20.f))
					{
						bClear = false;
						break;
					}
				}
				if (!bClear)
				{
					continue;
				}
				HazardOffsets[i] = Cand;
				bPlaced = true;
				break;
			}
			if (!bPlaced)
			{
				return false;
			}
		}
		return true;
	};

	for (int32 Attempt = 0; Attempt < 80; ++Attempt)
	{
		if (!ChooseGoal())
		{
			continue;
		}
		if (PlaceOnCorridor())
		{
			return true;
		}
	}

	// Fallback: fixed goal + one on-center hazard, others offset but still overlapping the path.
	GoalOffset = FVector2D(FMath::Max(MinClearance * 2.f, GoalRange * 0.6f), 0.f);
	const FVector2D PathDir = GoalOffset.GetSafeNormal();
	const FVector2D Perp(-PathDir.Y, PathDir.X);
	const float Straddle = FMath::Min(CorridorHalfWidth, BlockHalf * 0.6f);
	for (int32 i = 0; i < N; ++i)
	{
		const float T = 0.25f + 0.5f * (static_cast<float>(i) + 0.5f) / static_cast<float>(N);
		const float Lateral = (i == 0) ? 0.f : ((i % 2 == 0) ? Straddle : -Straddle);
		HazardOffsets[i] = GoalOffset * T + Perp * Lateral;
	}
	return false;
}

rlmesh::ResetResult URLMeshAvoidHazardsComponent::ResetEnv(
	std::optional<int64_t> Seed, const std::optional<rlmesh::InfoDict>& /*Options*/)
{
	if (Seed)
	{
		const uint64 Bits = static_cast<uint64>(*Seed);
		Random.Initialize(static_cast<int32>(Bits ^ (Bits >> 32)));
	}
	AActor* Owner = GetOwner();
	Owner->SetActorLocation(Start, false, nullptr, ETeleportType::TeleportPhysics);
	EnsureHazardActors();
	const bool bSampled = SampleLayout();
	Steps = 0;
	bLastHitHazard = false;
	bLastSuccess = false;
	SyncGoalActor();
	SyncHazardActors();
	const FVector Loc = Owner->GetActorLocation();
	UE_LOG(LogRLMesh, Log,
		   TEXT("Avoid Hazards reset: %s at (%.1f, %.1f) goal (%.1f, %.1f) hazards=%d layout=%s"),
		   *Owner->GetName(), Loc.X, Loc.Y, Start.X + GoalOffset.X, Start.Y + GoalOffset.Y,
		   FMath::Max(1, NumHazards), bSampled ? TEXT("ok") : TEXT("fallback"));
	return {Observe(), MakeInfo(false, false)};
}

rlmesh::StepResult URLMeshAvoidHazardsComponent::StepEnv(const std::optional<rlmesh::Value>& Action)
{
	if (!Action)
	{
		throw rlmesh::RecoverableEnvironmentException(
			"step needs a float32 action of shape (2,) in [-1, 1]");
	}

	FVector2D Dir = AvoidHazardsDetail::ActionAsDirectionXY(*Action);
	AActor* Owner = GetOwner();
	const FVector2D RemBefore = RemainingGoalXY();
	const double DistBefore = RemBefore.Size();
	if (!Dir.IsNearlyZero())
	{
		Dir.Normalize();
		double Travel = static_cast<double>(StepSize);
		// When closing on the goal, don't overshoot past it (was oscillating forever).
		if (DistBefore > 1e-6 && FVector2D::DotProduct(Dir, RemBefore) > 0.0)
		{
			Travel = FMath::Min(Travel, DistBefore);
		}
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
	const bool bHit = OverlapsHazard();
	const double Dist = DistanceGoalXY();
	const bool bReached = Dist <= FMath::Max(1.0, static_cast<double>(StepSize) * 0.5);
	bLastHitHazard = bHit;
	bLastSuccess = bReached && !bHit;

	const double RewardScale = FMath::Max(static_cast<double>(GoalRange), 1.0);
	double Reward = -Dist / RewardScale;
	if (bHit)
	{
		Reward -= HazardPenalty;
	}

	const bool bTerminated = bHit || bReached;
	const bool bTruncated = !bTerminated && Steps >= MaxSteps;
	const FVector Loc = Owner->GetActorLocation();
	UE_LOG(LogRLMesh, Log,
		   TEXT("Avoid Hazards step %d: dist=%.1f hit=%d success=%d -> (%.1f, %.1f)%s"), Steps, Dist,
		   bHit ? 1 : 0, bLastSuccess ? 1 : 0, Loc.X, Loc.Y,
		   bTruncated ? TEXT(" truncated") : TEXT(""));
	return {Observe(), Reward, bTerminated, bTruncated, MakeInfo(bHit, bLastSuccess)};
}

FVector2D URLMeshAvoidHazardsComponent::RemainingGoalXY() const
{
	const FVector Loc = GetOwner()->GetActorLocation();
	return FVector2D(Start.X + GoalOffset.X - Loc.X, Start.Y + GoalOffset.Y - Loc.Y);
}

double URLMeshAvoidHazardsComponent::DistanceGoalXY() const
{
	return RemainingGoalXY().Size();
}

bool URLMeshAvoidHazardsComponent::OverlapsHazard() const
{
	const FVector Loc = GetOwner()->GetActorLocation();
	const FVector2D AgentXY(Loc.X, Loc.Y);
	const int32 N = FMath::Min(HazardOffsets.Num(), FMath::Max(1, NumHazards));
	for (int32 i = 0; i < N; ++i)
	{
		const FVector2D HazardXY(Start.X + HazardOffsets[i].X, Start.Y + HazardOffsets[i].Y);
		if (AvoidHazardsDetail::AabbOverlapXY(AgentXY, AgentHalfExtent, HazardXY, HazardHalfExtent))
		{
			return true;
		}
	}
	return false;
}

rlmesh::Value URLMeshAvoidHazardsComponent::Observe() const
{
	const int32 Dim = ObsDim();
	std::vector<float> Observation(static_cast<size_t>(Dim), 0.f);
	const FVector2D GoalRem = RemainingGoalXY();
	Observation[0] = static_cast<float>(GoalRem.X);
	Observation[1] = static_cast<float>(GoalRem.Y);
	const FVector Loc = GetOwner()->GetActorLocation();
	const int32 N = FMath::Max(1, NumHazards);
	for (int32 i = 0; i < N; ++i)
	{
		FVector2D HazardWorld = FVector2D::ZeroVector;
		if (HazardOffsets.IsValidIndex(i))
		{
			HazardWorld = FVector2D(Start.X + HazardOffsets[i].X, Start.Y + HazardOffsets[i].Y);
		}
		Observation[static_cast<size_t>(2 + 2 * i)] = static_cast<float>(HazardWorld.X - Loc.X);
		Observation[static_cast<size_t>(2 + 2 * i + 1)] = static_cast<float>(HazardWorld.Y - Loc.Y);
	}
	std::vector<size_t> Shape = {static_cast<size_t>(Dim)};
	return rlmesh::Tensor(Observation.data(), Observation.size() * sizeof(float), std::move(Shape),
						  rlmesh::DType::Float32);
}

rlmesh::InfoDict URLMeshAvoidHazardsComponent::MakeInfo(bool bHitHazard, bool bSuccess) const
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
	Info.insert_or_assign("distance", DistanceGoalXY());
	Info.insert_or_assign("steps", static_cast<int64_t>(Steps));
	Info.insert_or_assign("hit_hazard", static_cast<int64_t>(bHitHazard ? 1 : 0));
	Info.insert_or_assign("success", static_cast<int64_t>(bSuccess ? 1 : 0));
	Info.insert_or_assign("num_hazards", static_cast<int64_t>(FMath::Max(1, NumHazards)));
	return Info;
}
