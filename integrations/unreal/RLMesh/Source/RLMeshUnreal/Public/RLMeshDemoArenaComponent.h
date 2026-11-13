#pragma once

#include "CoreMinimal.h"
#include "RLMeshEnvComponent.h"

#include "RLMeshDemoArenaComponent.generated.h"

/**
 * Level-driven arena env: collect labeled Coins and the Goal
 * (goal is treated like a final coin), fail on labeled Obstacles, stop against
 * walls/floor via swept movement.
 *
 * Observation is Box[2 + 2*NumCoins + 2*NumObstacles + NumWallRays]: goal XY
 * gap (zeros once collected), per-coin XY gaps (zeros if collected),
 * per-obstacle XY gaps, then NumWallRays rangefinder distances (uu; index 0 =
 * +X, then CCW). Action is Box([-1, 1]^2).
 *
 * Reward is -dist_nearest_remaining_target / GoalRange +1 per coin or goal
 * pickup. Success when every coin and the goal are collected; failure on
 * obstacle overlap; truncates after MaxSteps.
 *
 * Coins/goal use non-blocking collision so the agent can overlap them; on
 * pickup they hide and stay non-blocking. Does not spawn layout props or write
 * materials — binds actors by label.
 */
UCLASS(ClassGroup = (RLMesh), meta = (BlueprintSpawnableComponent))
class RLMESHUNREAL_API URLMeshDemoArenaComponent : public URLMeshEnvComponent
{
	GENERATED_BODY()

public:
	URLMeshDemoArenaComponent();

	UPROPERTY(EditAnywhere, Category = "RLMesh", meta = (ClampMin = "1"))
	float StepSize = 50.f;

	/** Scales the distance-shaping term (nearest remaining coin/goal). */
	UPROPERTY(EditAnywhere, Category = "RLMesh", meta = (ClampMin = "1"))
	float GoalRange = 2000.f;

	UPROPERTY(EditAnywhere, Category = "RLMesh", meta = (ClampMin = "1"))
	int32 MaxSteps = 400;

	UPROPERTY(EditAnywhere, Category = "RLMesh", meta = (ClampMin = "0", ClampMax = "16"))
	int32 NumCoins = 3;

	UPROPERTY(EditAnywhere, Category = "RLMesh", meta = (ClampMin = "0", ClampMax = "16"))
	int32 NumObstacles = 1;

	/** XY radius (uu) for coin/goal pickup; also OR'd with AABB half-extents. */
	UPROPERTY(EditAnywhere, Category = "RLMesh", meta = (ClampMin = "1"))
	float PickupRadius = 55.f;

	UPROPERTY(EditAnywhere, Category = "RLMesh", meta = (ClampMin = "1"))
	float AgentHalfExtent = 25.f;

	UPROPERTY(EditAnywhere, Category = "RLMesh", meta = (ClampMin = "1"))
	float CoinHalfExtent = 30.f;

	UPROPERTY(EditAnywhere, Category = "RLMesh", meta = (ClampMin = "1"))
	float ObstacleHalfExtent = 50.f;

	/** Number of XY rangefinder rays appended to the observation (0 disables). */
	UPROPERTY(EditAnywhere, Category = "RLMesh", meta = (ClampMin = "0", ClampMax = "16"))
	int32 NumWallRays = 8;

	/** Max length (uu) of each wall rangefinder ray. */
	UPROPERTY(EditAnywhere, Category = "RLMesh", meta = (ClampMin = "1"))
	float WallSenseRange = 350.f;

	/**
	 * When true, StepEnv updates the logical pose immediately (obs/pickups) while
	 * the rendered actor eases from the previous logical pose over MovementInterpTime
	 * (about one step of visual lag — stays correct at high Serving Max FPS).
	 */
	UPROPERTY(EditAnywhere, Category = "RLMesh")
	bool bInterpolateMovement = true;

	/** Seconds to ease the visual mesh toward the logical step pose. */
	UPROPERTY(EditAnywhere, Category = "RLMesh",
			  meta = (EditCondition = "bInterpolateMovement", ClampMin = "0.01", ClampMax = "1.0"))
	float MovementInterpTime = 0.08f;

	/**
	 * Optional level Goal actor. When empty, BeginPlay/Reset binds the first
	 * actor whose label starts with "Goal".
	 */
	UPROPERTY(EditAnywhere, Category = "RLMesh")
	TObjectPtr<AActor> GoalActor;

protected:
	virtual void BeginPlay() override;
	virtual void TickComponent(float DeltaTime, ELevelTick TickType,
							   FActorComponentTickFunction* ThisTickFunction) override;

	virtual rlmesh::spaces::Space ObservationSpace() const override;
	virtual rlmesh::spaces::Space ActionSpace() const override;
	virtual rlmesh::ResetResult ResetEnv(std::optional<int64_t> Seed,
										 const std::optional<rlmesh::InfoDict>& Options) override;
	virtual rlmesh::StepResult StepEnv(const std::optional<rlmesh::Value>& Action) override;

private:
	int32 ObsDim() const;
	void EnsureAgentCollision() const;
	void EnsurePickupProp(AActor* Actor) const;
	void LockOwnerZ();
	void BindLevelActors();
	void CaptureCoinSpawns();
	void RestoreCoins();
	void RestoreGoal();
	FVector AgentLocation() const;
	void SetLogicalLocation(const FVector& Loc, bool bSweepMove);
	void TickMovementInterp(float DeltaTime);
	FVector2D RemainingGoalXY() const;
	double DistanceGoalXY() const;
	FVector2D RemainingNearestTargetXY() const;
	double DistanceNearestTargetXY() const;
	bool OverlapsObstacle() const;
	bool OverlapsPickupXY(const FVector& TargetLoc) const;
	int32 TryCollectCoins();
	bool TryCollectGoal();
	int32 CoinsLeft() const;
	int32 TargetsLeft() const;
	void SampleWallRanges(float* OutRanges, int32 Count) const;
	rlmesh::Value Observe() const;
	rlmesh::InfoDict MakeInfo(bool bHitObstacle, bool bSuccess, int32 CollectedThisStep) const;

	FVector Start = FVector::ZeroVector;
	FVector LogicalLocation = FVector::ZeroVector;
	FVector VisualFrom = FVector::ZeroVector;
	float MovementInterpAlpha = 1.f;
	FVector2D GoalOffset = FVector2D::ZeroVector;
	bool bHaveGoal = false;
	bool bGoalCollected = false;

	TArray<TObjectPtr<AActor>> CoinActors;
	TArray<FVector> CoinSpawnWorld;
	TArray<uint8> CoinCollected;

	TArray<TObjectPtr<AActor>> ObstacleActors;

	int32 Steps = 0;
	int32 TotalCollected = 0;
	bool bLastHitObstacle = false;
	bool bLastSuccess = false;
	bool bBound = false;
};
