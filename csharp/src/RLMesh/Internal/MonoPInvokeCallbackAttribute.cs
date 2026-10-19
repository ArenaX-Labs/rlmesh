using System;

namespace AOT;

// IL2CPP recognizes the attribute's name. It is binding plumbing, never public.
[AttributeUsage(AttributeTargets.Method)]
internal sealed class MonoPInvokeCallbackAttribute : Attribute
{
    internal MonoPInvokeCallbackAttribute(Type delegateType) { }
}
