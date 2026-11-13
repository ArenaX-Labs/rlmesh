#include "Modules/ModuleManager.h"
#include "RLMeshEnvComponent.h"

class FRLMeshUnrealModule : public IModuleInterface
{
public:
	virtual void StartupModule() override
	{
		// Pin HostIdentity before any Play / EnvServer so Serving #keys stay stable.
		URLMeshEnvComponent::ApplyDefaultHostIdentity();
	}
};

IMPLEMENT_MODULE(FRLMeshUnrealModule, RLMeshUnreal)
