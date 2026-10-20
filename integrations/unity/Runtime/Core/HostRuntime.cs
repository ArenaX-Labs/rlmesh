using System;
using System.Globalization;
using System.Text;

namespace RLMesh.Unity
{
    public static class HostRuntime
    {
        public static string Json(string unityVersion, bool il2cpp, string backendVersion)
        {
            return "{\"language\":\"csharp\",\"framework_versions\":{\"unity\":" + Quote(unityVersion)
                + ",\"" + (il2cpp ? "il2cpp" : "mono") + "\":" + Quote(backendVersion) + "}}";
        }

        private static string Quote(string value)
        {
            if (value == null) throw new ArgumentNullException(nameof(value));
            var result = new StringBuilder("\"");
            foreach (char c in value)
            {
                if (c == '"' || c == '\\') result.Append('\\').Append(c);
                else if (c < 32 || char.IsSurrogate(c)) result.Append("\\u").Append(((int)c).ToString("x4", CultureInfo.InvariantCulture));
                else result.Append(c);
            }
            return result.Append('"').ToString();
        }
    }
}
