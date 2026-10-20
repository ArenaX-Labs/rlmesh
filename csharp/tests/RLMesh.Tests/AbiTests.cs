using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using RLMesh.Interop;
using Xunit;

namespace RLMesh.Tests;

public sealed unsafe class AbiTests
{
    [Fact]
    public void EveryGeneratedLayoutCompilesAsCOnAllFour64BitTargets()
    {
        var source = new StringBuilder("#include \"rlmesh.h\"\n");
        var layouts = new HashSet<Type>();
        int count = 0;
        foreach (Type type in typeof(Native).Assembly.GetTypes().Where(t => t.Namespace == "RLMesh.Interop" && t.IsValueType && !t.IsEnum && !t.IsNested))
        {
            var fields = type.GetFields(BindingFlags.Instance | BindingFlags.Public);
            if (fields.Length == 0) continue; // C opaque forward declarations have no layout.
            layouts.Add(type);
            string name = type.Name == "DLManagedTensorVersioned" ? "struct DLManagedTensorVersioned" : type.Name;
            source.AppendLine($"_Static_assert(sizeof({name}) == {Marshal.SizeOf(type)}, \"{name} size\");");
            foreach (var field in fields)
            {
                source.AppendLine($"_Static_assert(offsetof({name}, {field.Name}) == {Marshal.OffsetOf(type, field.Name)}, \"{name}.{field.Name}\");");
                // Fixed buffers and the bound union are checked as complete fields.
                int size = field.FieldType.IsPointer || field.FieldType == typeof(IntPtr) || field.FieldType == typeof(UIntPtr) ? 8 : field.FieldType.IsEnum ? 4 : Marshal.SizeOf(field.FieldType);
                source.AppendLine($"_Static_assert(sizeof((({name}*)0)->{field.Name}) == {size}, \"{name}.{field.Name} size\");");
            }
            count++;
        }
        source.AppendLine($"_Static_assert(RLMESH_ABI_VERSION == {Initializers.AbiVersion}, \"ABI generation\");");
        Assert.True(count > 40, $"only {count} struct layouts checked");
        Assert.Contains(typeof(RLMeshEnvFactoryServeOptions), layouts);
        Assert.Contains(typeof(RLMeshExtEnvFactoryServeOptions), layouts);
        Assert.Contains(typeof(RLMeshExtModelServeOptions), layouts);
        string file = Path.Combine(FoundationTests.Root, "target/csharp-layout.c");
        File.WriteAllText(file, source.ToString());
        foreach (string triple in new[] { "x86_64-linux-gnu", "x86_64-windows-msvc", "x86_64-macos", "aarch64-macos" })
        {
            var result = FoundationTests.Process("zig", new[] { "cc", "-target", triple, "-std=c11", "-ffreestanding", "-c", "-I", "crates/rlmesh-capi/include", file, "-o", Path.Combine(FoundationTests.Root, "target/csharp-layout-" + triple + ".o") });
            Assert.True(result.Code == 0, triple + ": " + result.Output);
        }
    }
    [Fact]
    public void ManagedInitializersEqualCFieldByField()
    {
        var output = FoundationTests.Process(Path.Combine(FoundationTests.Root, "target/csharp-init"), Array.Empty<string>());
        Assert.Equal(0, output.Code);
        using var expected = JsonDocument.Parse(output.Output);
        var cases = new Dictionary<string, object>
        {
            ["build"] = Initializers.BuildInfo(), ["box"] = Initializers.BoxOptions(),
            ["discrete"] = Initializers.DiscreteOptions(), ["serve"] = Initializers.ServeOptions(),
            ["server"] = Initializers.EnvServerOptions(), ["ext"] = Initializers.ExtEnvServerOptions((sbyte*)1),
            ["attributes"] = Initializers.EnvAttributes(), ["vtable"] = Initializers.EnvVtable(),
            ["factory"] = Initializers.EnvFactoryServeOptions(),
            ["factory_ext"] = Initializers.ExtEnvFactoryServeOptions((sbyte*)1),
            ["model_ext"] = Initializers.ExtModelServeOptions(),
        };
        foreach (var pair in cases)
        {
            var fields = pair.Value.GetType().GetFields(BindingFlags.Public | BindingFlags.Instance);
            var native = expected.RootElement.GetProperty(pair.Key);
            Assert.Equal(fields.Length, native.EnumerateObject().Count());
            foreach (var field in fields)
            {
                var exp = native.GetProperty(field.Name);
                object? value = field.GetValue(pair.Value);
                if (field.FieldType.IsPointer) Assert.Equal(exp.GetBoolean(), Pointer.Unbox(value!) != null);
                else if (field.FieldType == typeof(IntPtr)) Assert.Equal(exp.GetBoolean(), (IntPtr)value! != IntPtr.Zero);
                else if (field.FieldType == typeof(UIntPtr)) Assert.Equal(exp.GetUInt64(), ((UIntPtr)value!).ToUInt64());
                else if (field.FieldType.IsEnum) Assert.Equal(exp.GetInt64(), Convert.ToInt64(value));
                else if (field.FieldType == typeof(double)) Assert.Equal(exp.GetDouble(), (double)value!);
                else if (field.GetCustomAttribute<FixedBufferAttribute>() is { } buffer)
                {
                    Assert.True(exp.GetBoolean());
                    IntPtr memory = Marshal.AllocHGlobal(Marshal.SizeOf(pair.Value));
                    try
                    {
                        Marshal.StructureToPtr(pair.Value, memory, false);
                        for (int i = 0; i < buffer.Length; i++) Assert.Equal(0, Marshal.ReadByte(memory, Marshal.OffsetOf(pair.Value.GetType(), field.Name).ToInt32() + i));
                    }
                    finally { Marshal.FreeHGlobal(memory); }
                }
                else Assert.Equal(exp.GetInt64(), Convert.ToInt64(value));
            }
        }
        Assert.Equal(Initializers.PackageVersion, expected.RootElement.GetProperty("version").GetString());
    }
}
