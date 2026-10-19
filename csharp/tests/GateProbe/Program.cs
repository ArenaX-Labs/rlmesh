using System;
using RLMesh;
using RLMesh.Spaces;
using System.Collections.Generic;

for (int i = 0; i < 2; i++)
{
    try
    {
        if (args.Length > 0) _ = new Dict(Array.Empty<KeyValuePair<string, Space>>());
        else _ = Value.None;
        return 1;
    }
    catch (RuntimeError error) { Console.WriteLine(System.Text.Json.JsonSerializer.Serialize(error.Message)); }
}
return 0;
