#include "RLMeshReachTargetComponent.h"

#include "Components/SceneComponent.h"
#include "GameFramework/Actor.h"

#include <limits>

void URLMeshReachTargetComponent::BeginPlay()
{
	AActor* Owner = GetOwner();
	// Steps move the actor, which a Static root refuses.
	USceneComponent* Root = Owner->GetRootComponent();
	if (Root && Root->Mobility == EComponentMobility::Static)
	{
		Root->SetMobility(EComponentMobility::Movable);
		UE_LOG(LogRLMesh, Log, TEXT("Reach Target: set %s to Movable so steps can move it"), *Owner->GetName());
	}
	// Remember the start before Super::BeginPlay starts serving.
	Start = Owner->GetActorLocation();
	Random.GenerateNewSeed();
	Super::BeginPlay();
}

rlmesh::spaces::Space URLMeshReachTargetComponent::ObservationSpace() const
{
	const double Infinity = std::numeric_limits<double>::infinity();
	return rlmesh::spaces::Box(-Infinity, Infinity, {1});
}

rlmesh::spaces::Space URLMeshReachTargetComponent::ActionSpace() const
{
	return rlmesh::spaces::Discrete(3);
}

rlmesh::ResetResult URLMeshReachTargetComponent::ResetEnv(std::optional<int64_t> Seed,
														  const std::optional<rlmesh::InfoDict>& /*Options*/)
{
	if (Seed)
	{
		// FRandomStream takes an int32: fold in the seed's high half so seeds
		// that differ only there still pick different targets.
		const uint64 Bits = static_cast<uint64>(*Seed);
		Random.Initialize(static_cast<int32>(Bits ^ (Bits >> 32)));
	}
	GetOwner()->SetActorLocation(Start);
	Target = Random.FRandRange(-TargetRange, TargetRange);
	Steps = 0;
	return {Observe(), {}};
}

rlmesh::StepResult URLMeshReachTargetComponent::StepEnv(const std::optional<rlmesh::Value>& Action)
{
	if (!Action)
	{
		throw rlmesh::RecoverableEnvironmentException("step needs an action: 0, 1 or 2");
	}
	const int64_t Move = Action->as_int() - 1;
	GetOwner()->AddActorWorldOffset(FVector(static_cast<double>(Move) * StepSize, 0.0, 0.0));
	++Steps;

	const double Distance = FMath::Abs(Offset());
	const bool bReached = Distance <= StepSize / 2.0;
	return {Observe(), -Distance / TargetRange, bReached, !bReached && Steps >= MaxSteps, {}};
}

double URLMeshReachTargetComponent::Offset() const
{
	return Target - (GetOwner()->GetActorLocation().X - Start.X);
}

rlmesh::Value URLMeshReachTargetComponent::Observe() const
{
	const float Observation = static_cast<float>(Offset());
	return rlmesh::Tensor(&Observation, sizeof(Observation), {1}, rlmesh::DType::Float32);
}
