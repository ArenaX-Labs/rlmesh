#pragma once

#include "CoreMinimal.h"
#include "Math/RandomStream.h"
#include "Materials/MaterialInterface.h"
#include "RLMeshEnvComponent.h"

#include "RLMeshCollectCoinsComponent.generated.h"

/**
 * Example env: slide the owning actor on the XY plane and collect N yellow
 * coin spheres.
 *
 * Observation is Box[3]: remaining XY gap to the nearest uncollected coin, then
 * coins_left / NumCoins. Action is Box([-1, 1]^2). Reward is +1 per pickup,
 * minus normalized distance to the nearest coin each step, and +1 when the last
 * coin is collected. Terminates when all coins are collected; truncates after
 * MaxSteps. info carries success / coins_left / collected.
 */
UCLASS(ClassGroup = (RLMesh), meta = (BlueprintSpawnableComponent))
class RLMESHUNREAL_API URLMeshCollectCoinsComponent : public URLMeshEnvComponent
{
	GENERATED_BODY()

public:
	URLMeshCollectCoinsComponent();

	/** Distance one step moves the agent on the plane, in units. */
	UPROPERTY(EditAnywhere, Category = "RLMesh", meta = (ClampMin = "1"))
	float StepSize = 50.f;

	/** Farthest a random coin is placed from the start (XY). Scales shaping. */
	UPROPERTY(EditAnywhere, Category = "RLMesh", meta = (ClampMin = "1"))
	float CoinRange = 1000.f;

	/** Steps before an episode truncates. */
	UPROPERTY(EditAnywhere, Category = "RLMesh", meta = (ClampMin = "1"))
	int32 MaxSteps = 400;

	/** Number of coin spheres (fixed for the episode). */
	UPROPERTY(EditAnywhere, Category = "RLMesh", meta = (ClampMin = "1", ClampMax = "16"))
	int32 NumCoins = 3;

	/** XY pickup radius (agent center to coin center). */
	UPROPERTY(EditAnywhere, Category = "RLMesh", meta = (ClampMin = "1"))
	float PickupRadius = 40.f;

	/** Minimum spacing between coins and from the start. */
	UPROPERTY(EditAnywhere, Category = "RLMesh", meta = (ClampMin = "1"))
	float MinClearance = 200.f;

	/** Visual half-extent of each coin sphere (BasicShapes Sphere scale). */
	UPROPERTY(EditAnywhere, Category = "RLMesh", meta = (ClampMin = "1"))
	float CoinHalfExtent = 30.f;

	/** Optional material for runtime-spawned coin spheres. */
	UPROPERTY(EditAnywhere, Category = "RLMesh")
	TObjectPtr<UMaterialInterface> CoinMaterial;

protected:
	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

	virtual rlmesh::spaces::Space ObservationSpace() const override;
	virtual rlmesh::spaces::Space ActionSpace() const override;
	virtual rlmesh::ResetResult ResetEnv(std::optional<int64_t> Seed,
										 const std::optional<rlmesh::InfoDict>& Options) override;
	virtual rlmesh::StepResult StepEnv(const std::optional<rlmesh::Value>& Action) override;

private:
	void LockOwnerZ() const;
	void EnsureCoinActors();
	void DestroyCoinActors();
	void SyncCoinActors() const;
	bool SampleLayout();
	int32 CoinsLeft() const;
	int32 NearestCoinIndex() const;
	FVector2D RemainingNearestCoinXY() const;
	double DistanceNearestCoinXY() const;
	int32 TryCollect();
	rlmesh::Value Observe() const;
	rlmesh::InfoDict MakeInfo(bool bSuccess, int32 CollectedThisStep) const;

	FVector Start = FVector::ZeroVector;
	TArray<FVector2D> CoinOffsets;
	TArray<uint8> CoinCollected;
	TArray<TObjectPtr<AActor>> CoinActors;
	int32 Steps = 0;
	int32 TotalCollected = 0;
	bool bLastSuccess = false;
	FRandomStream Random;
};
