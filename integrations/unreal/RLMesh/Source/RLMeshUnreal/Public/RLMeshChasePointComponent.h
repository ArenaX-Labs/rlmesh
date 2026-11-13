#pragma once

#include "CoreMinimal.h"
#include "Math/RandomStream.h"
#include "RLMeshEnvComponent.h"

#include "RLMeshChasePointComponent.generated.h"

/**
 * Example env: slide the owning actor on the XY plane toward a moving 2D goal.
 *
 * Each reset puts the actor back where play began (Z locked thereafter). If
 * GoalActor is set, the goal starts at that actor's world XY; otherwise a goal
 * offset is picked in [-GoalRange, GoalRange]^2. For the first GoalMoveSteps
 * of the episode the goal wanders randomly on the floor (not toward the agent,
 * clamped inside GoalWanderRadius of its home). Then it stops so the agent can
 * catch up. Observation is the remaining XY gap (Box[2]). Action is
 * Box([-1, 1]^2). Reward is minus XY distance over GoalRange; terminates near
 * the goal; truncates after MaxSteps.
 */
UCLASS(ClassGroup = (RLMesh), meta = (BlueprintSpawnableComponent))
class RLMESHUNREAL_API URLMeshChasePointComponent : public URLMeshEnvComponent
{
	GENERATED_BODY()

public:
	URLMeshChasePointComponent();

	/** Distance one step moves the agent on the plane, in units. */
	UPROPERTY(EditAnywhere, Category = "RLMesh", meta = (ClampMin = "1"))
	float StepSize = 50.f;

	/** Farthest a random goal is placed from the start (XY). Also scales reward. */
	UPROPERTY(EditAnywhere, Category = "RLMesh", meta = (ClampMin = "1"))
	float GoalRange = 1000.f;

	/** Steps before an episode truncates. */
	UPROPERTY(EditAnywhere, Category = "RLMesh", meta = (ClampMin = "1"))
	int32 MaxSteps = 300;

	/**
	 * How many env steps the goal keeps wandering after reset.
	 * After this, the goal freezes for the rest of the episode.
	 */
	UPROPERTY(EditAnywhere, Category = "RLMesh", meta = (ClampMin = "0"))
	int32 GoalMoveSteps = 40;

	/** Distance the goal moves on each wandering step, in units. */
	UPROPERTY(EditAnywhere, Category = "RLMesh", meta = (ClampMin = "0"))
	float GoalStepSize = 35.f;

	/**
	 * Goal stays inside this XY radius of its reset home (on the floor plane).
	 * Prevents walking off the playable disk.
	 */
	UPROPERTY(EditAnywhere, Category = "RLMesh", meta = (ClampMin = "1"))
	float GoalWanderRadius = 600.f;

	/**
	 * Optional level actor used as the XY goal (e.g. a Goal sphere).
	 * When set, reset uses this actor's world XY and moves it while wandering.
	 */
	UPROPERTY(EditAnywhere, Category = "RLMesh")
	TObjectPtr<AActor> GoalActor;

protected:
	virtual void BeginPlay() override;

	virtual rlmesh::spaces::Space ObservationSpace() const override;
	virtual rlmesh::spaces::Space ActionSpace() const override;
	virtual rlmesh::ResetResult ResetEnv(std::optional<int64_t> Seed,
										 const std::optional<rlmesh::InfoDict>& Options) override;
	virtual rlmesh::StepResult StepEnv(const std::optional<rlmesh::Value>& Action) override;

private:
	FVector2D RemainingXY() const;
	double DistanceXY() const;
	rlmesh::Value Observe() const;
	rlmesh::InfoDict MakeInfo() const;
	void LockOwnerZ() const;
	void SyncGoalActor() const;
	void WanderGoal();

	FVector Start = FVector::ZeroVector;
	/** Goal as world-space XY offset from Start (Z ignored). */
	FVector2D GoalOffset = FVector2D::ZeroVector;
	/** XY home (world) used to clamp wandering for this episode. */
	FVector2D GoalHome = FVector2D::ZeroVector;
	/** Fixed spawn offset when GoalActor is set (so resets don't drift). */
	FVector2D GoalSpawnOffset = FVector2D::ZeroVector;
	bool bHaveGoalSpawn = false;
	int32 Steps = 0;
	FRandomStream Random;
};
