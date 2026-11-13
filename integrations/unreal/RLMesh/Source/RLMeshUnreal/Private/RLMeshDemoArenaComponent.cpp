#include "RLMeshDemoArenaComponent.h"

#include "Components/SceneComponent.h"
#include "Components/StaticMeshComponent.h"
#include "Engine/World.h"
#include "EngineUtils.h"
#include "GameFramework/Actor.h"

#include <cstddef>
#include <limits>
#include <vector>

namespace DemoArenaDetail
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

bool LabelStartsWith(const AActor* Actor, const TCHAR* Prefix)
{
	if (!IsValid(Actor) || !Prefix)
	{
		return false;
	}
	const FString Label = Actor->GetActorLabel().ToLower();
	return Label.StartsWith(FString(Prefix).ToLower());
}

bool ClassNameContains(const AActor* Actor, const TCHAR* Token)
{
	if (!IsValid(Actor) || !Token)
	{
		return false;
	}
	const FString Name = Actor->GetClass()->GetName().ToLower();
	return Name.Contains(FString(Token).ToLower());
}

bool MatchesRole(const AActor* Actor, const TCHAR* LabelPrefix, const TCHAR* ClassToken)
{
	return LabelStartsWith(Actor, LabelPrefix) || ClassNameContains(Actor, ClassToken);
}
} // namespace DemoArenaDetail

URLMeshDemoArenaComponent::URLMeshDemoArenaComponent()
{
	EnvId = TEXT("Unreal-DemoArena-v0");
}

int32 URLMeshDemoArenaComponent::ObsDim() const
{
	return 2 + 2 * FMath::Max(0, NumCoins) + 2 * FMath::Max(0, NumObstacles) +
		   FMath::Clamp(NumWallRays, 0, 16);
}

void URLMeshDemoArenaComponent::EnsureAgentCollision() const
{
	AActor* Owner = GetOwner();
	if (!Owner)
	{
		return;
	}
	if (USceneComponent* Root = Owner->GetRootComponent())
	{
		if (Root->Mobility == EComponentMobility::Static)
		{
			Root->SetMobility(EComponentMobility::Movable);
			UE_LOG(LogRLMesh, Log,
				   TEXT("Demo Arena: set %s root mobility to Movable so steps can move it"),
				   *Owner->GetName());
		}
	}
	if (UStaticMeshComponent* SMC = Owner->FindComponentByClass<UStaticMeshComponent>())
	{
		SMC->SetMobility(EComponentMobility::Movable);
		SMC->SetSimulatePhysics(false);
		SMC->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
		SMC->SetCollisionProfileName(TEXT("BlockAll"));
	}
}

void URLMeshDemoArenaComponent::BeginPlay()
{
	AActor* Owner = GetOwner();
	if (Owner)
	{
		Start = Owner->GetActorLocation();
		LogicalLocation = Start;
		VisualFrom = Start;
		MovementInterpAlpha = 1.f;
	}
	EnsureAgentCollision();
	BindLevelActors();
	CaptureCoinSpawns();
	PrimaryComponentTick.bCanEverTick = true;
	Super::BeginPlay();
}

void URLMeshDemoArenaComponent::TickComponent(float DeltaTime, ELevelTick TickType,
											  FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
	TickMovementInterp(DeltaTime);
}

FVector URLMeshDemoArenaComponent::AgentLocation() const
{
	return LogicalLocation;
}

void URLMeshDemoArenaComponent::SetLogicalLocation(const FVector& Loc, bool bSweepMove)
{
	AActor* Owner = GetOwner();
	if (!Owner)
	{
		return;
	}
	// Sweep/collide from the previous *logical* pose — never from the lagging mesh.
	// At high lockstep rates the display can trail many steps; using it as the sweep
	// origin would pin the sim against walls while pickups still resolve from LogicalLocation.
	const FVector PrevLogical = LogicalLocation;
	FVector Resolved = Loc;
	Resolved.Z = Start.Z;
	if (bSweepMove)
	{
		Owner->SetActorLocation(PrevLogical, /*bSweep=*/false, nullptr, ETeleportType::TeleportPhysics);
		FHitResult Hit;
		Owner->SetActorLocation(Resolved, /*bSweep=*/true, &Hit, ETeleportType::TeleportPhysics);
		if (Hit.bBlockingHit && Hit.ImpactNormal.Z > 0.7f)
		{
			Owner->SetActorLocation(Resolved, /*bSweep=*/false, nullptr, ETeleportType::TeleportPhysics);
		}
		else if (Hit.bBlockingHit)
		{
			// Wall: keep whatever sweep left us at for logical pose.
			Resolved = Owner->GetActorLocation();
			Resolved.Z = Start.Z;
			Owner->SetActorLocation(Resolved, false, nullptr, ETeleportType::TeleportPhysics);
		}
	}
	else
	{
		Owner->SetActorLocation(Resolved, false, nullptr, ETeleportType::TeleportPhysics);
	}
	LogicalLocation = Owner->GetActorLocation();
	LogicalLocation.Z = Start.Z;
	if (bInterpolateMovement)
	{
		// Ease only from the previous sim pose → at most ~one step of visual lag,
		// so a fast lockstep episode does not leave the mesh stranded at the start.
		VisualFrom = PrevLogical;
		VisualFrom.Z = Start.Z;
		MovementInterpAlpha = 0.f;
		Owner->SetActorLocation(VisualFrom, false, nullptr, ETeleportType::TeleportPhysics);
	}
	else
	{
		MovementInterpAlpha = 1.f;
		Owner->SetActorLocation(LogicalLocation, false, nullptr, ETeleportType::TeleportPhysics);
	}
}

void URLMeshDemoArenaComponent::TickMovementInterp(float DeltaTime)
{
	if (!bInterpolateMovement || MovementInterpAlpha >= 1.f)
	{
		return;
	}
	AActor* Owner = GetOwner();
	if (!Owner)
	{
		return;
	}
	const float Duration = FMath::Max(MovementInterpTime, 0.01f);
	MovementInterpAlpha = FMath::Clamp(MovementInterpAlpha + DeltaTime / Duration, 0.f, 1.f);
	const FVector Visual = FMath::Lerp(VisualFrom, LogicalLocation, MovementInterpAlpha);
	Owner->SetActorLocation(Visual, false, nullptr, ETeleportType::TeleportPhysics);
}

void URLMeshDemoArenaComponent::BindLevelActors()
{
	UWorld* World = GetWorld();
	if (!World)
	{
		return;
	}

	CoinActors.Reset();
	ObstacleActors.Reset();

	AActor* FoundGoal = IsValid(GoalActor) ? GoalActor.Get() : nullptr;

	for (TActorIterator<AActor> It(World); It; ++It)
	{
		AActor* Actor = *It;
		if (!IsValid(Actor) || Actor == GetOwner())
		{
			continue;
		}
		if (!FoundGoal && DemoArenaDetail::MatchesRole(Actor, TEXT("Goal"), TEXT("Goal")))
		{
			FoundGoal = Actor;
			continue;
		}
		if (DemoArenaDetail::MatchesRole(Actor, TEXT("Coin"), TEXT("Coin")) &&
			CoinActors.Num() < NumCoins)
		{
			CoinActors.Add(Actor);
			continue;
		}
		const bool bObstacle =
			DemoArenaDetail::MatchesRole(Actor, TEXT("Obstacle"), TEXT("Obstacle")) ||
			DemoArenaDetail::MatchesRole(Actor, TEXT("Hazard"), TEXT("Hazard"));
		if (bObstacle && ObstacleActors.Num() < NumObstacles)
		{
			ObstacleActors.Add(Actor);
		}
	}

	GoalActor = FoundGoal;
	if (IsValid(GoalActor))
	{
		const FVector GoalLoc = GoalActor->GetActorLocation();
		GoalOffset = FVector2D(GoalLoc.X - Start.X, GoalLoc.Y - Start.Y);
		bHaveGoal = true;
		if (USceneComponent* GoalRoot = GoalActor->GetRootComponent())
		{
			if (GoalRoot->Mobility == EComponentMobility::Static)
			{
				GoalRoot->SetMobility(EComponentMobility::Movable);
			}
		}
		EnsurePickupProp(GoalActor);
	}
	else
	{
		bHaveGoal = false;
		UE_LOG(LogRLMesh, Error,
			   TEXT("Demo Arena: no Goal actor found (label starting with Goal) and GoalActor unset"));
	}

	for (AActor* Coin : CoinActors)
	{
		EnsurePickupProp(Coin);
	}

	if (CoinActors.Num() < NumCoins)
	{
		UE_LOG(LogRLMesh, Warning,
			   TEXT("Demo Arena: found %d coin actor(s), expected NumCoins=%d (labels Coin*)"),
			   CoinActors.Num(), NumCoins);
	}
	if (ObstacleActors.Num() < NumObstacles)
	{
		UE_LOG(LogRLMesh, Warning,
			   TEXT("Demo Arena: found %d obstacle actor(s), expected NumObstacles=%d "
					"(labels Obstacle*/Hazard*)"),
			   ObstacleActors.Num(), NumObstacles);
	}

	bBound = true;
	UE_LOG(LogRLMesh, Log,
		   TEXT("Demo Arena bind: goal=%s coins=%d obstacles=%d"),
		   IsValid(GoalActor) ? *GoalActor->GetActorLabel() : TEXT("(none)"), CoinActors.Num(),
		   ObstacleActors.Num());
}

void URLMeshDemoArenaComponent::EnsurePickupProp(AActor* Actor) const
{
	if (!IsValid(Actor))
	{
		return;
	}
	// Pickup props must not block swept steps — agent overlaps them to collect.
	Actor->SetActorEnableCollision(false);
	if (UStaticMeshComponent* SMC = Actor->FindComponentByClass<UStaticMeshComponent>())
	{
		SMC->SetCollisionEnabled(ECollisionEnabled::NoCollision);
		SMC->SetGenerateOverlapEvents(false);
	}
}

void URLMeshDemoArenaComponent::CaptureCoinSpawns()
{
	CoinSpawnWorld.SetNum(CoinActors.Num());
	CoinCollected.SetNum(CoinActors.Num());
	for (int32 i = 0; i < CoinActors.Num(); ++i)
	{
		CoinCollected[i] = 0;
		if (IsValid(CoinActors[i]))
		{
			EnsurePickupProp(CoinActors[i]);
			CoinSpawnWorld[i] = CoinActors[i]->GetActorLocation();
		}
		else
		{
			CoinSpawnWorld[i] = Start;
		}
	}
}

void URLMeshDemoArenaComponent::RestoreCoins()
{
	const int32 N = CoinActors.Num();
	CoinCollected.SetNum(N);
	for (int32 i = 0; i < N; ++i)
	{
		CoinCollected[i] = 0;
		AActor* Coin = CoinActors[i];
		if (!IsValid(Coin))
		{
			continue;
		}
		const FVector Spawn =
			CoinSpawnWorld.IsValidIndex(i) ? CoinSpawnWorld[i] : Coin->GetActorLocation();
		Coin->SetActorLocation(Spawn, false, nullptr, ETeleportType::TeleportPhysics);
		Coin->SetActorHiddenInGame(false);
		EnsurePickupProp(Coin);
	}
}

void URLMeshDemoArenaComponent::RestoreGoal()
{
	bGoalCollected = false;
	if (!IsValid(GoalActor))
	{
		return;
	}
	GoalActor->SetActorHiddenInGame(false);
	EnsurePickupProp(GoalActor);
	const FVector GoalLoc = GoalActor->GetActorLocation();
	GoalOffset = FVector2D(GoalLoc.X - Start.X, GoalLoc.Y - Start.Y);
	bHaveGoal = true;
}

rlmesh::spaces::Space URLMeshDemoArenaComponent::ObservationSpace() const
{
	const double Infinity = std::numeric_limits<double>::infinity();
	return rlmesh::spaces::Box(-Infinity, Infinity, {ObsDim()});
}

rlmesh::spaces::Space URLMeshDemoArenaComponent::ActionSpace() const
{
	return rlmesh::spaces::Box(-1.0, 1.0, {2});
}

void URLMeshDemoArenaComponent::LockOwnerZ()
{
	LogicalLocation.Z = Start.Z;
}

FVector2D URLMeshDemoArenaComponent::RemainingGoalXY() const
{
	if (bGoalCollected || !bHaveGoal)
	{
		return FVector2D::ZeroVector;
	}
	const FVector Loc = AgentLocation();
	return FVector2D(Start.X + GoalOffset.X - Loc.X, Start.Y + GoalOffset.Y - Loc.Y);
}

double URLMeshDemoArenaComponent::DistanceGoalXY() const
{
	return RemainingGoalXY().Size();
}

FVector2D URLMeshDemoArenaComponent::RemainingNearestTargetXY() const
{
	FVector2D Best = RemainingGoalXY();
	double BestDist = bGoalCollected || !bHaveGoal ? std::numeric_limits<double>::infinity()
												   : Best.Size();

	const FVector Loc = AgentLocation();
	const int32 CoinSlots = FMath::Max(0, NumCoins);
	for (int32 i = 0; i < CoinSlots; ++i)
	{
		if (CoinCollected.IsValidIndex(i) && CoinCollected[i] != 0)
		{
			continue;
		}
		if (!CoinActors.IsValidIndex(i) || !IsValid(CoinActors[i]))
		{
			continue;
		}
		const FVector CLoc = CoinActors[i]->GetActorLocation();
		const FVector2D Gap(CLoc.X - Loc.X, CLoc.Y - Loc.Y);
		const double Dist = Gap.Size();
		if (Dist < BestDist)
		{
			BestDist = Dist;
			Best = Gap;
		}
	}
	if (BestDist > 1.0e300)
	{
		return FVector2D::ZeroVector;
	}
	return Best;
}

double URLMeshDemoArenaComponent::DistanceNearestTargetXY() const
{
	return RemainingNearestTargetXY().Size();
}

bool URLMeshDemoArenaComponent::OverlapsPickupXY(const FVector& TargetLoc) const
{
	const FVector Loc = AgentLocation();
	const FVector2D AgentXY(Loc.X, Loc.Y);
	const FVector2D TargetXY(TargetLoc.X, TargetLoc.Y);
	const float Radius = FMath::Max(PickupRadius, StepSize * 0.55f);
	if ((TargetXY - AgentXY).Size() <= Radius)
	{
		return true;
	}
	return DemoArenaDetail::AabbOverlapXY(AgentXY, AgentHalfExtent, TargetXY, CoinHalfExtent);
}

bool URLMeshDemoArenaComponent::OverlapsObstacle() const
{
	const FVector Loc = AgentLocation();
	const FVector2D AgentXY(Loc.X, Loc.Y);
	for (const TObjectPtr<AActor>& Obstacle : ObstacleActors)
	{
		if (!IsValid(Obstacle))
		{
			continue;
		}
		const FVector OLoc = Obstacle->GetActorLocation();
		if (DemoArenaDetail::AabbOverlapXY(AgentXY, AgentHalfExtent,
										   FVector2D(OLoc.X, OLoc.Y), ObstacleHalfExtent))
		{
			return true;
		}
	}
	return false;
}

int32 URLMeshDemoArenaComponent::CoinsLeft() const
{
	int32 Left = 0;
	for (uint8 Flag : CoinCollected)
	{
		if (Flag == 0)
		{
			++Left;
		}
	}
	return Left;
}

int32 URLMeshDemoArenaComponent::TargetsLeft() const
{
	return CoinsLeft() + ((bHaveGoal && !bGoalCollected) ? 1 : 0);
}

int32 URLMeshDemoArenaComponent::TryCollectCoins()
{
	int32 Collected = 0;
	for (int32 i = 0; i < CoinActors.Num(); ++i)
	{
		if (CoinCollected.IsValidIndex(i) && CoinCollected[i] != 0)
		{
			continue;
		}
		AActor* Coin = CoinActors[i];
		if (!IsValid(Coin))
		{
			continue;
		}
		if (OverlapsPickupXY(Coin->GetActorLocation()))
		{
			CoinCollected[i] = 1;
			Coin->SetActorHiddenInGame(true);
			EnsurePickupProp(Coin);
			++Collected;
			++TotalCollected;
			UE_LOG(LogRLMesh, Verbose, TEXT("Demo Arena: collected coin[%d] (%s)"), i,
				   *Coin->GetActorLabel());
		}
	}
	return Collected;
}

bool URLMeshDemoArenaComponent::TryCollectGoal()
{
	if (!bHaveGoal || bGoalCollected || !IsValid(GoalActor))
	{
		return false;
	}
	const FVector GoalLoc(Start.X + GoalOffset.X, Start.Y + GoalOffset.Y, Start.Z);
	if (!OverlapsPickupXY(GoalLoc))
	{
		return false;
	}
	bGoalCollected = true;
	GoalActor->SetActorHiddenInGame(true);
	EnsurePickupProp(GoalActor);
	++TotalCollected;
	UE_LOG(LogRLMesh, Verbose, TEXT("Demo Arena: collected goal (%s)"), *GoalActor->GetActorLabel());
	return true;
}

rlmesh::ResetResult URLMeshDemoArenaComponent::ResetEnv(
	std::optional<int64_t> /*Seed*/, const std::optional<rlmesh::InfoDict>& /*Options*/)
{
	if (!bBound)
	{
		BindLevelActors();
		CaptureCoinSpawns();
	}

	AActor* Owner = GetOwner();
	EnsureAgentCollision();
	SetLogicalLocation(Start, /*bSweepMove=*/false);
	MovementInterpAlpha = 1.f;
	if (Owner)
	{
		Owner->SetActorLocation(LogicalLocation, false, nullptr, ETeleportType::TeleportPhysics);
	}

	RestoreGoal();
	RestoreCoins();
	Steps = 0;
	TotalCollected = 0;
	bLastHitObstacle = false;
	bLastSuccess = false;

	const FVector Loc = AgentLocation();
	UE_LOG(LogRLMesh, Log,
		   TEXT("Demo Arena reset: %s at (%.1f, %.1f) goal=(%.1f, %.1f) coins=%d obstacles=%d "
				"targets=%d"),
		   *Owner->GetName(), Loc.X, Loc.Y, Start.X + GoalOffset.X, Start.Y + GoalOffset.Y,
		   CoinActors.Num(), ObstacleActors.Num(), TargetsLeft());
	return {Observe(), MakeInfo(false, false, 0)};
}

rlmesh::StepResult URLMeshDemoArenaComponent::StepEnv(const std::optional<rlmesh::Value>& Action)
{
	if (!Action)
	{
		throw rlmesh::RecoverableEnvironmentException(
			"step needs a float32 action of shape (2,) in [-1, 1]");
	}

	FVector2D Dir = DemoArenaDetail::ActionAsDirectionXY(*Action);
	const FVector2D RemBefore = RemainingNearestTargetXY();
	const double DistBefore = RemBefore.Size();

	if (!Dir.IsNearlyZero())
	{
		Dir.Normalize();
		double Travel = static_cast<double>(StepSize);
		// Snap onto the active target so the agent can overlap/collect it.
		if (DistBefore > 1e-6 && FVector2D::DotProduct(Dir, RemBefore) > 0.0)
		{
			Travel = FMath::Min(Travel, DistBefore);
		}
		const FVector Loc = AgentLocation();
		const FVector Desired(Loc.X + Dir.X * Travel, Loc.Y + Dir.Y * Travel, Start.Z);
		SetLogicalLocation(Desired, /*bSweepMove=*/true);
		LockOwnerZ();
	}
	else
	{
		LockOwnerZ();
	}

	++Steps;
	const bool bHit = OverlapsObstacle();
	int32 CollectedThisStep = 0;
	if (!bHit)
	{
		CollectedThisStep += TryCollectCoins();
		if (TryCollectGoal())
		{
			++CollectedThisStep;
		}
	}
	const double Dist = DistanceNearestTargetXY();
	const bool bAllDone = TargetsLeft() == 0;
	bLastHitObstacle = bHit;
	bLastSuccess = bAllDone && !bHit;

	const double RewardScale = FMath::Max(static_cast<double>(GoalRange), 1.0);
	double Reward = -Dist / RewardScale;
	Reward += static_cast<double>(CollectedThisStep);

	const bool bTerminated = bHit || bAllDone;
	const bool bTruncated = !bTerminated && Steps >= MaxSteps;

	if (bTerminated || bTruncated)
	{
		const FVector Loc = AgentLocation();
		UE_LOG(LogRLMesh, Log,
			   TEXT("Demo Arena step %d: nearest=%.1f hit=%d coins_left=%d goal_left=%d picked=%d "
					"success=%d -> (%.1f, %.1f)%s"),
			   Steps, Dist, bHit ? 1 : 0, CoinsLeft(), (bHaveGoal && !bGoalCollected) ? 1 : 0,
			   CollectedThisStep, bLastSuccess ? 1 : 0, Loc.X, Loc.Y,
			   bTruncated ? TEXT(" truncated") : TEXT(""));
	}
	else if (CollectedThisStep > 0 || (Steps % 50) == 0)
	{
		const FVector Loc = AgentLocation();
		UE_LOG(LogRLMesh, Verbose,
			   TEXT("Demo Arena step %d: nearest=%.1f hit=%d coins_left=%d goal_left=%d picked=%d "
					"-> (%.1f, %.1f)"),
			   Steps, Dist, bHit ? 1 : 0, CoinsLeft(), (bHaveGoal && !bGoalCollected) ? 1 : 0,
			   CollectedThisStep, Loc.X, Loc.Y);
	}

	return {Observe(), Reward, bTerminated, bTruncated,
			MakeInfo(bHit, bLastSuccess, CollectedThisStep)};
}

void URLMeshDemoArenaComponent::SampleWallRanges(float* OutRanges, int32 Count) const
{
	if (!OutRanges || Count <= 0)
	{
		return;
	}
	const float Range = FMath::Max(WallSenseRange, 1.f);
	for (int32 i = 0; i < Count; ++i)
	{
		OutRanges[i] = Range;
	}

	AActor* Owner = GetOwner();
	UWorld* World = GetWorld();
	if (!Owner || !World)
	{
		return;
	}

	const FVector Origin = AgentLocation();
	FCollisionQueryParams Params(SCENE_QUERY_STAT(DemoArenaWallSense), false, Owner);
	Params.bReturnPhysicalMaterial = false;
	const float Radius = FMath::Clamp(AgentHalfExtent * 0.5f, 5.f, 25.f);
	const FCollisionShape SenseShape = FCollisionShape::MakeSphere(Radius);
	FCollisionObjectQueryParams ObjectParams;
	ObjectParams.AddObjectTypesToQuery(ECC_WorldStatic);
	ObjectParams.AddObjectTypesToQuery(ECC_WorldDynamic);
	ObjectParams.AddObjectTypesToQuery(ECC_PhysicsBody);

	for (int32 i = 0; i < Count; ++i)
	{
		const float Angle = (2.f * PI) * (static_cast<float>(i) / static_cast<float>(Count));
		const FVector Dir(FMath::Cos(Angle), FMath::Sin(Angle), 0.f);
		const FVector End = Origin + Dir * Range;
		FHitResult Hit;
		if (!World->SweepSingleByObjectType(Hit, Origin, End, FQuat::Identity, ObjectParams,
											SenseShape, Params))
		{
			continue;
		}
		// Floor hits (mostly upward normals) are not walls.
		if (Hit.ImpactNormal.Z > 0.7f)
		{
			continue;
		}
		OutRanges[i] = FMath::Clamp(Hit.Distance, 0.f, Range);
	}
}

rlmesh::Value URLMeshDemoArenaComponent::Observe() const
{
	const int32 Dim = ObsDim();
	std::vector<float> Observation(static_cast<size_t>(Dim), 0.f);
	const FVector2D GoalRem = RemainingGoalXY(); // zeros once goal is collected
	Observation[0] = static_cast<float>(GoalRem.X);
	Observation[1] = static_cast<float>(GoalRem.Y);

	const FVector Loc = AgentLocation();
	int32 Idx = 2;
	const int32 CoinSlots = FMath::Max(0, NumCoins);
	for (int32 i = 0; i < CoinSlots; ++i)
	{
		float Dx = 0.f;
		float Dy = 0.f;
		if (CoinActors.IsValidIndex(i) && IsValid(CoinActors[i]) &&
			(!CoinCollected.IsValidIndex(i) || CoinCollected[i] == 0))
		{
			const FVector CLoc = CoinActors[i]->GetActorLocation();
			Dx = static_cast<float>(CLoc.X - Loc.X);
			Dy = static_cast<float>(CLoc.Y - Loc.Y);
		}
		Observation[static_cast<size_t>(Idx++)] = Dx;
		Observation[static_cast<size_t>(Idx++)] = Dy;
	}

	const int32 ObstacleSlots = FMath::Max(0, NumObstacles);
	for (int32 i = 0; i < ObstacleSlots; ++i)
	{
		float Dx = 0.f;
		float Dy = 0.f;
		if (ObstacleActors.IsValidIndex(i) && IsValid(ObstacleActors[i]))
		{
			const FVector OLoc = ObstacleActors[i]->GetActorLocation();
			Dx = static_cast<float>(OLoc.X - Loc.X);
			Dy = static_cast<float>(OLoc.Y - Loc.Y);
		}
		Observation[static_cast<size_t>(Idx++)] = Dx;
		Observation[static_cast<size_t>(Idx++)] = Dy;
	}

	const int32 WallSlots = FMath::Clamp(NumWallRays, 0, 16);
	if (WallSlots > 0 && Idx + WallSlots <= Dim)
	{
		SampleWallRanges(Observation.data() + Idx, WallSlots);
		Idx += WallSlots;
	}
	(void)Idx;

	std::vector<size_t> Shape = {static_cast<size_t>(Dim)};
	return rlmesh::Tensor(Observation.data(), Observation.size() * sizeof(float), std::move(Shape),
						  rlmesh::DType::Float32);
}

rlmesh::InfoDict URLMeshDemoArenaComponent::MakeInfo(bool bHitObstacle, bool bSuccess,
													 int32 CollectedThisStep) const
{
	const FVector Loc = AgentLocation();
	const float AgentXy[2] = {static_cast<float>(Loc.X), static_cast<float>(Loc.Y)};
	const float GoalXy[2] = {static_cast<float>(Start.X + GoalOffset.X),
							 static_cast<float>(Start.Y + GoalOffset.Y)};
	rlmesh::InfoDict Info;
	Info.insert_or_assign(
		"agent_xy", rlmesh::Tensor(AgentXy, sizeof(AgentXy), {2}, rlmesh::DType::Float32));
	Info.insert_or_assign(
		"goal_xy", rlmesh::Tensor(GoalXy, sizeof(GoalXy), {2}, rlmesh::DType::Float32));
	Info.insert_or_assign("distance", DistanceNearestTargetXY());
	Info.insert_or_assign("distance_goal", DistanceGoalXY());
	Info.insert_or_assign("coins_left", static_cast<int64_t>(CoinsLeft()));
	Info.insert_or_assign("goal_left", static_cast<int64_t>((bHaveGoal && !bGoalCollected) ? 1 : 0));
	Info.insert_or_assign("targets_left", static_cast<int64_t>(TargetsLeft()));
	Info.insert_or_assign("collected", static_cast<int64_t>(CollectedThisStep));
	Info.insert_or_assign("total_collected", static_cast<int64_t>(TotalCollected));
	Info.insert_or_assign("hit_obstacle", static_cast<int64_t>(bHitObstacle ? 1 : 0));
	Info.insert_or_assign("success", static_cast<int64_t>(bSuccess ? 1 : 0));
	Info.insert_or_assign("steps", static_cast<int64_t>(Steps));
	Info.insert_or_assign("num_coins", static_cast<int64_t>(FMath::Max(0, NumCoins)));
	Info.insert_or_assign("num_obstacles", static_cast<int64_t>(FMath::Max(0, NumObstacles)));
	Info.insert_or_assign("num_wall_rays",
						  static_cast<int64_t>(FMath::Clamp(NumWallRays, 0, 16)));
	Info.insert_or_assign("wall_sense_range", static_cast<double>(WallSenseRange));
	return Info;
}
