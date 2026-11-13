#include "RLMeshCollectCoinsComponent.h"

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

namespace CollectCoinsDetail
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
} // namespace CollectCoinsDetail

URLMeshCollectCoinsComponent::URLMeshCollectCoinsComponent()
{
	EnvId = TEXT("Unreal-CollectCoins-v0");
}

void URLMeshCollectCoinsComponent::BeginPlay()
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
					   TEXT("Collect Coins: set %s root mobility to Movable so steps can move it"),
					   *Owner->GetName());
			}
		}
		Start = Owner->GetActorLocation();
	}
	Random.GenerateNewSeed();
	EnsureCoinActors();
	Super::BeginPlay();
}

void URLMeshCollectCoinsComponent::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	DestroyCoinActors();
	Super::EndPlay(EndPlayReason);
}

rlmesh::spaces::Space URLMeshCollectCoinsComponent::ObservationSpace() const
{
	const double Infinity = std::numeric_limits<double>::infinity();
	return rlmesh::spaces::Box(-Infinity, Infinity, {3});
}

rlmesh::spaces::Space URLMeshCollectCoinsComponent::ActionSpace() const
{
	return rlmesh::spaces::Box(-1.0, 1.0, {2});
}

void URLMeshCollectCoinsComponent::LockOwnerZ() const
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

void URLMeshCollectCoinsComponent::EnsureCoinActors()
{
	UWorld* World = GetWorld();
	if (!World)
	{
		return;
	}
	const int32 N = FMath::Max(1, NumCoins);
	CoinOffsets.SetNum(N);
	CoinCollected.SetNum(N);
	while (CoinActors.Num() > N)
	{
		AActor* Extra = CoinActors.Pop();
		if (IsValid(Extra))
		{
			Extra->Destroy();
		}
	}
	UStaticMesh* Mesh =
		LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Sphere.Sphere"));
	const float Scale = (CoinHalfExtent * 2.f) / 100.f; // BasicShapes Sphere is 100uu
	while (CoinActors.Num() < N)
	{
		FActorSpawnParameters Params;
		Params.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
		AStaticMeshActor* Coin =
			World->SpawnActor<AStaticMeshActor>(Start, FRotator::ZeroRotator, Params);
		if (!Coin)
		{
			break;
		}
		Coin->SetActorLabel(FString::Printf(TEXT("Coin_%d"), CoinActors.Num()));
		if (UStaticMeshComponent* SMC = Coin->GetStaticMeshComponent())
		{
			SMC->SetMobility(EComponentMobility::Movable);
			if (Mesh)
			{
				SMC->SetStaticMesh(Mesh);
			}
			SMC->SetWorldScale3D(FVector(Scale));
			SMC->SetCollisionEnabled(ECollisionEnabled::NoCollision);
			if (CoinMaterial)
			{
				SMC->SetMaterial(0, CoinMaterial);
			}
		}
		CoinActors.Add(Coin);
	}
}

void URLMeshCollectCoinsComponent::DestroyCoinActors()
{
	for (AActor* Coin : CoinActors)
	{
		if (IsValid(Coin))
		{
			Coin->Destroy();
		}
	}
	CoinActors.Reset();
}

void URLMeshCollectCoinsComponent::SyncCoinActors() const
{
	const int32 N = FMath::Min(CoinActors.Num(), CoinOffsets.Num());
	for (int32 i = 0; i < N; ++i)
	{
		AActor* Coin = CoinActors[i];
		if (!IsValid(Coin))
		{
			continue;
		}
		const bool bTaken = CoinCollected.IsValidIndex(i) && CoinCollected[i] != 0;
		const FVector2D& Off = CoinOffsets[i];
		Coin->SetActorLocation(FVector(Start.X + Off.X, Start.Y + Off.Y, Start.Z), false, nullptr,
							   ETeleportType::TeleportPhysics);
		Coin->SetActorHiddenInGame(bTaken);
		Coin->SetActorEnableCollision(false);
	}
}

bool URLMeshCollectCoinsComponent::SampleLayout()
{
	const int32 N = FMath::Max(1, NumCoins);
	CoinOffsets.SetNum(N);
	CoinCollected.SetNum(N);
	for (int32 i = 0; i < N; ++i)
	{
		CoinCollected[i] = 0;
	}

	auto FarEnough = [this](const FVector2D& A, const FVector2D& B) -> bool
	{
		return (A - B).Size() >= MinClearance;
	};

	for (int32 Attempt = 0; Attempt < 80; ++Attempt)
	{
		bool bOk = true;
		for (int32 i = 0; i < N; ++i)
		{
			bool bPlaced = false;
			for (int32 Inner = 0; Inner < 48; ++Inner)
			{
				const FVector2D Cand(Random.FRandRange(-CoinRange, CoinRange),
									 Random.FRandRange(-CoinRange, CoinRange));
				if (!FarEnough(Cand, FVector2D::ZeroVector))
				{
					continue;
				}
				bool bClear = true;
				for (int32 j = 0; j < i; ++j)
				{
					if (!FarEnough(Cand, CoinOffsets[j]))
					{
						bClear = false;
						break;
					}
				}
				if (!bClear)
				{
					continue;
				}
				CoinOffsets[i] = Cand;
				bPlaced = true;
				break;
			}
			if (!bPlaced)
			{
				bOk = false;
				break;
			}
		}
		if (bOk)
		{
			return true;
		}
	}

	// Fallback: ring around the start.
	for (int32 i = 0; i < N; ++i)
	{
		const float Angle = (2.f * PI * static_cast<float>(i)) / static_cast<float>(N);
		const float Radius = FMath::Max(MinClearance, CoinRange * 0.45f);
		CoinOffsets[i] = FVector2D(FMath::Cos(Angle) * Radius, FMath::Sin(Angle) * Radius);
		CoinCollected[i] = 0;
	}
	return false;
}

int32 URLMeshCollectCoinsComponent::CoinsLeft() const
{
	int32 Left = 0;
	const int32 N = FMath::Min(CoinCollected.Num(), FMath::Max(1, NumCoins));
	for (int32 i = 0; i < N; ++i)
	{
		if (CoinCollected[i] == 0)
		{
			++Left;
		}
	}
	return Left;
}

int32 URLMeshCollectCoinsComponent::NearestCoinIndex() const
{
	const FVector Loc = GetOwner()->GetActorLocation();
	const FVector2D AgentXY(Loc.X, Loc.Y);
	int32 Best = INDEX_NONE;
	double BestDist = std::numeric_limits<double>::infinity();
	const int32 N = FMath::Min(CoinOffsets.Num(), FMath::Max(1, NumCoins));
	for (int32 i = 0; i < N; ++i)
	{
		if (CoinCollected.IsValidIndex(i) && CoinCollected[i] != 0)
		{
			continue;
		}
		const FVector2D CoinXY(Start.X + CoinOffsets[i].X, Start.Y + CoinOffsets[i].Y);
		const double Dist = (CoinXY - AgentXY).Size();
		if (Dist < BestDist)
		{
			BestDist = Dist;
			Best = i;
		}
	}
	return Best;
}

FVector2D URLMeshCollectCoinsComponent::RemainingNearestCoinXY() const
{
	const int32 Idx = NearestCoinIndex();
	if (Idx == INDEX_NONE)
	{
		return FVector2D::ZeroVector;
	}
	const FVector Loc = GetOwner()->GetActorLocation();
	return FVector2D(Start.X + CoinOffsets[Idx].X - Loc.X, Start.Y + CoinOffsets[Idx].Y - Loc.Y);
}

double URLMeshCollectCoinsComponent::DistanceNearestCoinXY() const
{
	return RemainingNearestCoinXY().Size();
}

int32 URLMeshCollectCoinsComponent::TryCollect()
{
	const FVector Loc = GetOwner()->GetActorLocation();
	const FVector2D AgentXY(Loc.X, Loc.Y);
	const float Radius = FMath::Max(PickupRadius, StepSize * 0.5f);
	int32 Collected = 0;
	const int32 N = FMath::Min(CoinOffsets.Num(), FMath::Max(1, NumCoins));
	for (int32 i = 0; i < N; ++i)
	{
		if (CoinCollected.IsValidIndex(i) && CoinCollected[i] != 0)
		{
			continue;
		}
		const FVector2D CoinXY(Start.X + CoinOffsets[i].X, Start.Y + CoinOffsets[i].Y);
		if ((CoinXY - AgentXY).Size() <= Radius)
		{
			CoinCollected[i] = 1;
			++Collected;
			++TotalCollected;
		}
	}
	if (Collected > 0)
	{
		SyncCoinActors();
	}
	return Collected;
}

rlmesh::ResetResult URLMeshCollectCoinsComponent::ResetEnv(
	std::optional<int64_t> Seed, const std::optional<rlmesh::InfoDict>& /*Options*/)
{
	if (Seed)
	{
		const uint64 Bits = static_cast<uint64>(*Seed);
		Random.Initialize(static_cast<int32>(Bits ^ (Bits >> 32)));
	}
	AActor* Owner = GetOwner();
	Owner->SetActorLocation(Start, false, nullptr, ETeleportType::TeleportPhysics);
	EnsureCoinActors();
	const bool bSampled = SampleLayout();
	Steps = 0;
	TotalCollected = 0;
	bLastSuccess = false;
	SyncCoinActors();
	const FVector Loc = Owner->GetActorLocation();
	UE_LOG(LogRLMesh, Log,
		   TEXT("Collect Coins reset: %s at (%.1f, %.1f) coins=%d layout=%s nearest=%.1f"),
		   *Owner->GetName(), Loc.X, Loc.Y, FMath::Max(1, NumCoins),
		   bSampled ? TEXT("ok") : TEXT("fallback"), DistanceNearestCoinXY());
	return {Observe(), MakeInfo(false, 0)};
}

rlmesh::StepResult URLMeshCollectCoinsComponent::StepEnv(const std::optional<rlmesh::Value>& Action)
{
	if (!Action)
	{
		throw rlmesh::RecoverableEnvironmentException(
			"step needs a float32 action of shape (2,) in [-1, 1]");
	}

	FVector2D Dir = CollectCoinsDetail::ActionAsDirectionXY(*Action);
	AActor* Owner = GetOwner();
	const FVector2D RemBefore = RemainingNearestCoinXY();
	const double DistBefore = RemBefore.Size();
	if (!Dir.IsNearlyZero())
	{
		Dir.Normalize();
		double Travel = static_cast<double>(StepSize);
		// When closing on the nearest coin, don't overshoot past its center.
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
	const int32 CollectedThisStep = TryCollect();
	const int32 Left = CoinsLeft();
	const bool bSuccess = Left == 0;
	bLastSuccess = bSuccess;

	const double Dist = DistanceNearestCoinXY();
	const double RewardScale = FMath::Max(static_cast<double>(CoinRange), 1.0);
	double Reward = -Dist / RewardScale;
	Reward += static_cast<double>(CollectedThisStep);
	if (bSuccess)
	{
		Reward += 1.0;
	}

	const bool bTerminated = bSuccess;
	const bool bTruncated = !bTerminated && Steps >= MaxSteps;
	const FVector Loc = Owner->GetActorLocation();
	UE_LOG(LogRLMesh, Log,
		   TEXT("Collect Coins step %d: nearest=%.1f left=%d picked=%d success=%d -> (%.1f, %.1f)%s"),
		   Steps, Dist, Left, CollectedThisStep, bSuccess ? 1 : 0, Loc.X, Loc.Y,
		   bTruncated ? TEXT(" truncated") : TEXT(""));
	return {Observe(), Reward, bTerminated, bTruncated, MakeInfo(bSuccess, CollectedThisStep)};
}

rlmesh::Value URLMeshCollectCoinsComponent::Observe() const
{
	const FVector2D Rem = RemainingNearestCoinXY();
	const int32 N = FMath::Max(1, NumCoins);
	const float LeftNorm = static_cast<float>(CoinsLeft()) / static_cast<float>(N);
	const float Observation[3] = {static_cast<float>(Rem.X), static_cast<float>(Rem.Y), LeftNorm};
	return rlmesh::Tensor(Observation, sizeof(Observation), {3}, rlmesh::DType::Float32);
}

rlmesh::InfoDict URLMeshCollectCoinsComponent::MakeInfo(bool bSuccess, int32 CollectedThisStep) const
{
	const FVector Loc = GetOwner()->GetActorLocation();
	const float AgentXy[2] = {static_cast<float>(Loc.X), static_cast<float>(Loc.Y)};
	const FVector2D NearestRem = RemainingNearestCoinXY();
	const float NearestXy[2] = {static_cast<float>(Loc.X + NearestRem.X),
								static_cast<float>(Loc.Y + NearestRem.Y)};
	rlmesh::InfoDict Info;
	Info.insert_or_assign(
		"agent_xy", rlmesh::Tensor(AgentXy, sizeof(AgentXy), {2}, rlmesh::DType::Float32));
	Info.insert_or_assign(
		"nearest_coin_xy",
		rlmesh::Tensor(NearestXy, sizeof(NearestXy), {2}, rlmesh::DType::Float32));
	Info.insert_or_assign("coins_left", static_cast<int64_t>(CoinsLeft()));
	Info.insert_or_assign("collected", static_cast<int64_t>(CollectedThisStep));
	Info.insert_or_assign("total_collected", static_cast<int64_t>(TotalCollected));
	Info.insert_or_assign("steps", static_cast<int64_t>(Steps));
	Info.insert_or_assign("success", static_cast<int64_t>(bSuccess ? 1 : 0));
	Info.insert_or_assign("num_coins", static_cast<int64_t>(FMath::Max(1, NumCoins)));
	return Info;
}
