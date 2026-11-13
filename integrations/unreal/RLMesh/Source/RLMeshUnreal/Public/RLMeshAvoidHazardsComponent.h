#pragma once

#include "CoreMinimal.h"
#include "Math/RandomStream.h"
#include "Materials/MaterialInterface.h"
#include "RLMeshEnvComponent.h"

#include "RLMeshAvoidHazardsComponent.generated.h"

/**
 * Example env: slide the owning actor on the XY plane toward a goal while
 * avoiding red hazard cubes.
 *
 * Observation is Box[2 + 2*NumHazards]: remaining XY gap to the goal, then XY
 * gap to each hazard center (fixed count). Action is Box([-1, 1]^2). Reward is
 * minus normalized goal distance each step, and -HazardPenalty on overlap.
 * Terminates on goal reach (success) or hazard hit (failure); truncates after
 * MaxSteps. info carries success / hit_hazard.
 */
UCLASS(ClassGroup = (RLMesh), meta = (BlueprintSpawnableComponent))
class RLMESHUNREAL_API URLMeshAvoidHazardsComponent : public URLMeshEnvComponent
{
	GENERATED_BODY()

public:
	URLMeshAvoidHazardsComponent();

	/** Distance one step moves the agent on the plane, in units. */
	UPROPERTY(EditAnywhere, Category = "RLMesh", meta = (ClampMin = "1"))
	float StepSize = 50.f;

	/** Farthest a random goal / hazard is placed from the start (XY). */
	UPROPERTY(EditAnywhere, Category = "RLMesh", meta = (ClampMin = "1"))
	float GoalRange = 1000.f;

	/** Steps before an episode truncates. */
	UPROPERTY(EditAnywhere, Category = "RLMesh", meta = (ClampMin = "1"))
	int32 MaxSteps = 300;

	/** Number of hazard cubes (fixed observation size). */
	UPROPERTY(EditAnywhere, Category = "RLMesh", meta = (ClampMin = "1", ClampMax = "16"))
	int32 NumHazards = 4;

	/** Half-extent of each hazard AABB on XY (cube scale ~1 → 50). */
	UPROPERTY(EditAnywhere, Category = "RLMesh", meta = (ClampMin = "1"))
	float HazardHalfExtent = 50.f;

	/** Agent XY half-extent for overlap tests. */
	UPROPERTY(EditAnywhere, Category = "RLMesh", meta = (ClampMin = "1"))
	float AgentHalfExtent = 25.f;

	/** Minimum clearance of hazards from the start and goal. */
	UPROPERTY(EditAnywhere, Category = "RLMesh", meta = (ClampMin = "1"))
	float MinClearance = 150.f;

	/**
	 * Hazards are placed along the start→goal corridor within this half-width
	 * so the straight path is blocked (must steer around).
	 */
	UPROPERTY(EditAnywhere, Category = "RLMesh", meta = (ClampMin = "0"))
	float CorridorHalfWidth = 120.f;

	/** Added to the step reward when the agent overlaps a hazard. */
	UPROPERTY(EditAnywhere, Category = "RLMesh")
	float HazardPenalty = 1.f;

	/**
	 * Optional level actor used as the XY goal.
	 * When set, reset uses this actor's world XY.
	 */
	UPROPERTY(EditAnywhere, Category = "RLMesh")
	TObjectPtr<AActor> GoalActor;

	/** Optional material for runtime-spawned hazard cubes. */
	UPROPERTY(EditAnywhere, Category = "RLMesh")
	TObjectPtr<UMaterialInterface> HazardMaterial;

protected:
	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

	virtual rlmesh::spaces::Space ObservationSpace() const override;
	virtual rlmesh::spaces::Space ActionSpace() const override;
	virtual rlmesh::ResetResult ResetEnv(std::optional<int64_t> Seed,
										 const std::optional<rlmesh::InfoDict>& Options) override;
	virtual rlmesh::StepResult StepEnv(const std::optional<rlmesh::Value>& Action) override;

private:
	int32 ObsDim() const { return 2 + 2 * FMath::Max(1, NumHazards); }
	FVector2D RemainingGoalXY() const;
	double DistanceGoalXY() const;
	bool OverlapsHazard() const;
	void EnsureHazardActors();
	void DestroyHazardActors();
	void SyncHazardActors() const;
	void SyncGoalActor() const;
	void LockOwnerZ() const;
	bool SampleLayout();
	rlmesh::Value Observe() const;
	rlmesh::InfoDict MakeInfo(bool bHitHazard, bool bSuccess) const;

	FVector Start = FVector::ZeroVector;
	FVector2D GoalOffset = FVector2D::ZeroVector;
	FVector2D GoalSpawnOffset = FVector2D::ZeroVector;
	bool bHaveGoalSpawn = false;
	TArray<FVector2D> HazardOffsets;
	TArray<TObjectPtr<AActor>> HazardActors;
	int32 Steps = 0;
	bool bLastHitHazard = false;
	bool bLastSuccess = false;
	FRandomStream Random;
};
