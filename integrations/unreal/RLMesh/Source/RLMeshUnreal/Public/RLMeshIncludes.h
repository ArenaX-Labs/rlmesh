// Includes rlmesh.hpp the way Unreal expects third-party headers. rlmesh.hpp
// saves and restores Unreal's `check`, `verify`, `TEXT` and similar macros
// itself; THIRD_PARTY_INCLUDES_START/END silence warnings Unreal raises to
// errors. Outside Unreal (the plugin's standalone tests) it is a plain include.
#pragma once

#if __has_include("CoreMinimal.h")
#include "CoreMinimal.h"
THIRD_PARTY_INCLUDES_START
#include <rlmesh.hpp>
THIRD_PARTY_INCLUDES_END
#else
#include <rlmesh.hpp>
#endif
