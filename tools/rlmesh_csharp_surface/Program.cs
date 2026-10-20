using System.Collections.Immutable;
using System.Reflection;
using System.Reflection.Metadata;
using System.Reflection.PortableExecutable;
using System.Text.Json;

if (args.Length != 2) throw new ArgumentException("usage: rlmesh_csharp_surface assembly.dll output.json");
using var stream = File.OpenRead(args[0]);
using var pe = new PEReader(stream);
var reader = pe.GetMetadataReader();
var types = new TypeNames(reader);
var surface = new SortedDictionary<string, object>(StringComparer.Ordinal);
var violations = new List<string>();
string Name(EntityHandle handle) => types.Name(handle);
string Text(StringHandle handle) => reader.GetString(handle);
string[] Attrs(CustomAttributeHandleCollection handles) => handles.Select(h =>
{
    var attribute = reader.GetCustomAttribute(h);
    string name = attribute.Constructor.Kind switch
    {
        HandleKind.MemberReference => Name(reader.GetMemberReference((MemberReferenceHandle)attribute.Constructor).Parent),
        HandleKind.MethodDefinition => Name(reader.GetMethodDefinition((MethodDefinitionHandle)attribute.Constructor).GetDeclaringType()),
        _ => "?",
    };
    return name + ":" + Convert.ToHexString(reader.GetBlobBytes(attribute.Value));
}).Order(StringComparer.Ordinal).ToArray();
object? Constant(ConstantHandle handle)
{
    if (handle.IsNil) return null;
    var c = reader.GetConstant(handle);
    var blob = reader.GetBlobReader(c.Value);
    return c.TypeCode switch
    {
        ConstantTypeCode.Boolean => blob.ReadBoolean(), ConstantTypeCode.Byte => blob.ReadByte(),
        ConstantTypeCode.SByte => blob.ReadSByte(), ConstantTypeCode.Int16 => blob.ReadInt16(),
        ConstantTypeCode.UInt16 => blob.ReadUInt16(), ConstantTypeCode.Int32 => blob.ReadInt32(),
        ConstantTypeCode.UInt32 => blob.ReadUInt32(), ConstantTypeCode.Int64 => blob.ReadInt64(),
        ConstantTypeCode.UInt64 => blob.ReadUInt64(), ConstantTypeCode.Single => blob.ReadSingle(),
        ConstantTypeCode.Double => blob.ReadDouble(), ConstantTypeCode.Char => (int)blob.ReadChar(),
        ConstantTypeCode.String => blob.ReadUTF16(blob.Length), _ => null,
    };
}
object[] Generics(GenericParameterHandleCollection handles) => handles.Select(h =>
{
    var p = reader.GetGenericParameter(h);
    return (object)new { name = Text(p.Name), flags = p.Attributes.ToString(), constraints = p.GetConstraints().Select(c => Name(reader.GetGenericParameterConstraint(c).Type)).ToArray(), attributes = Attrs(p.GetCustomAttributes()) };
}).ToArray();
bool VisibleMethod(MethodDefinition method) => (method.Attributes & MethodAttributes.MemberAccessMask) is MethodAttributes.Public or MethodAttributes.Family or MethodAttributes.FamORAssem;
bool VisibleType(TypeDefinition type)
{
    if ((type.Attributes & TypeAttributes.VisibilityMask) is not (TypeAttributes.Public or TypeAttributes.NestedPublic or TypeAttributes.NestedFamily or TypeAttributes.NestedFamORAssem)) return false;
    var parent = type.GetDeclaringType();
    return parent.IsNil || VisibleType(reader.GetTypeDefinition(parent));
}

foreach (var h in reader.TypeDefinitions)
{
    var t = reader.GetTypeDefinition(h);
    if (!VisibleType(t)) continue;
    string name = Name(h);
    string baseName = t.BaseType.IsNil ? "" : Name(t.BaseType);
    surface["T:" + name] = new
    {
        kind = (t.Attributes & TypeAttributes.Interface) != 0 ? "interface" : baseName == "System.Enum" ? "enum" : baseName == "System.ValueType" ? "struct" : "class",
        signature = t.Attributes.ToString(), @base = baseName,
        protocols = t.GetInterfaceImplementations().Select(i => Name(reader.GetInterfaceImplementation(i).Interface)).Order(StringComparer.Ordinal).ToArray(),
        generics = Generics(t.GetGenericParameters()), attributes = Attrs(t.GetCustomAttributes()),
    };
    foreach (var mh in t.GetMethods())
    {
        var m = reader.GetMethodDefinition(mh);
        if (!VisibleMethod(m)) continue;
        var sig = m.DecodeSignature(types, null);
        string methodName = Text(m.Name);
        string arity = sig.GenericParameterCount == 0 ? "" : "``" + sig.GenericParameterCount;
        string id = "M:" + name + "." + (methodName == ".ctor" ? "#ctor" : methodName) + arity + "(" + string.Join(",", sig.ParameterTypes) + ")";
        if (methodName is "op_Implicit" or "op_Explicit") id += "~" + sig.ReturnType;
        surface[id] = new
        {
            kind = "method", signature = sig.ReturnType, flags = m.Attributes.ToString(),
            @params = m.GetParameters().Select(ph =>
            {
                var p = reader.GetParameter(ph);
                return new { name = Text(p.Name), sequence = p.SequenceNumber, type = p.SequenceNumber == 0 ? sig.ReturnType : sig.ParameterTypes[p.SequenceNumber - 1], flags = p.Attributes.ToString(), @default = Constant(p.GetDefaultValue()), attributes = Attrs(p.GetCustomAttributes()) };
            }).ToArray(),
            generics = Generics(m.GetGenericParameters()), attributes = Attrs(m.GetCustomAttributes()),
        };
    }
    foreach (var ph in t.GetProperties())
    {
        var p = reader.GetPropertyDefinition(ph);
        var access = p.GetAccessors();
        bool get = !access.Getter.IsNil && VisibleMethod(reader.GetMethodDefinition(access.Getter));
        bool set = !access.Setter.IsNil && VisibleMethod(reader.GetMethodDefinition(access.Setter));
        if (!get && !set) continue;
        var sig = p.DecodeSignature(types, null);
        surface["P:" + name + "." + Text(p.Name)] = new { kind = "property", signature = sig.ReturnType, @params = sig.ParameterTypes, get, set, attributes = Attrs(p.GetCustomAttributes()) };
    }
    foreach (var fh in t.GetFields())
    {
        var f = reader.GetFieldDefinition(fh);
        if ((f.Attributes & FieldAttributes.FieldAccessMask) is not (FieldAttributes.Public or FieldAttributes.Family or FieldAttributes.FamORAssem)) continue;
        surface["F:" + name + "." + Text(f.Name)] = new { kind = baseName == "System.Enum" ? "enum_value" : "field", signature = f.DecodeSignature(types, null), flags = f.Attributes.ToString(), value = Constant(f.GetDefaultValue()), attributes = Attrs(f.GetCustomAttributes()) };
    }
}

bool net10 = reader.GetAssemblyDefinition().GetCustomAttributes().Any(h =>
{
    var a = reader.GetCustomAttribute(h);
    return a.Constructor.Kind == HandleKind.MemberReference &&
        Name(reader.GetMemberReference((MemberReferenceHandle)a.Constructor).Parent) == "System.Runtime.Versioning.TargetFrameworkAttribute" &&
        System.Text.Encoding.UTF8.GetString(reader.GetBlobBytes(a.Value)).Contains(".NETCoreApp,Version=v10.0", StringComparison.Ordinal);
});

// Check references, imports and reverse-call addresses without loading the assembly.
foreach (var h in reader.MemberReferences)
{
    var m = reader.GetMemberReference(h);
    string parent = Name(m.Parent), member = Text(m.Name);
    if (parent.StartsWith("System.Reflection.Emit", StringComparison.Ordinal) || parent.StartsWith("Microsoft.CSharp", StringComparison.Ordinal)
        || parent == "System.Activator" || (parent.StartsWith("System.Linq.Expressions", StringComparison.Ordinal) && member == "Compile")
        || (parent == "System.Type" && member is not ("GetTypeFromHandle" or "IsAssignableFrom" or "get_Name") && !(net10 && member == "get_Assembly")))
        violations.Add("Forbidden reference: " + parent + "." + member);
}
foreach (var h in reader.MethodDefinitions)
{
    var m = reader.GetMethodDefinition(h);
    if ((m.Attributes & MethodAttributes.PinvokeImpl) == 0) continue;
    var import = m.GetImport();
    string module = Text(reader.GetModuleReference(import.Module).Name);
    if (module != "rlmesh_capi" || (import.Attributes & MethodImportAttributes.CallingConventionMask) != MethodImportAttributes.CallingConventionCDecl)
        violations.Add("Invalid import: " + Text(m.Name));
    var sig = m.DecodeSignature(types, null);
    foreach (string type in sig.ParameterTypes.Append(sig.ReturnType))
        if (type is "System.String" or "System.Boolean" or "System.Text.StringBuilder" || Attrs(m.GetCustomAttributes()).Any(a => a.Contains("MarshalAs", StringComparison.Ordinal)))
            violations.Add("Non-blittable import: " + Text(m.Name));
}
// Decode IL instructions enough to find actual ldftn instructions, not matching
// operand bytes. OpCode.OperandType describes every operand in the BCL.
var opcodes = typeof(System.Reflection.Emit.OpCodes).GetFields(BindingFlags.Public | BindingFlags.Static)
    .Select(f => (System.Reflection.Emit.OpCode)f.GetValue(null)!).ToDictionary(o => unchecked((ushort)o.Value));
foreach (var h in reader.MethodDefinitions)
{
    var method = reader.GetMethodDefinition(h);
    if (method.RelativeVirtualAddress == 0) continue;
    var body = pe.GetMethodBody(method.RelativeVirtualAddress).GetILReader();
    EntityHandle lastFunction = default;
    while (body.RemainingBytes > 0)
    {
        ushort value = body.ReadByte();
        if (value == 0xfe) value = (ushort)(0xfe00 | body.ReadByte());
        var opcode = opcodes[value];
        int operand = 0;
        switch (opcode.OperandType)
        {
            case System.Reflection.Emit.OperandType.InlineNone: break;
            case System.Reflection.Emit.OperandType.ShortInlineI:
            case System.Reflection.Emit.OperandType.ShortInlineVar:
            case System.Reflection.Emit.OperandType.ShortInlineBrTarget: body.ReadByte(); break;
            case System.Reflection.Emit.OperandType.InlineVar: body.ReadUInt16(); break;
            case System.Reflection.Emit.OperandType.InlineI8:
            case System.Reflection.Emit.OperandType.InlineR: body.ReadInt64(); break;
            case System.Reflection.Emit.OperandType.InlineSwitch:
                int count = body.ReadInt32(); for (int i = 0; i < count; i++) body.ReadInt32(); break;
            default: operand = body.ReadInt32(); break;
        }
        if (opcode.Name == "ldftn") lastFunction = System.Reflection.Metadata.Ecma335.MetadataTokens.EntityHandle(operand);
        else if (opcode.Name == "newobj" && !lastFunction.IsNil)
        {
            var ctorHandle = System.Reflection.Metadata.Ecma335.MetadataTokens.EntityHandle(operand);
            TypeDefinitionHandle delegateType = default;
            if (ctorHandle.Kind == HandleKind.MethodDefinition) delegateType = reader.GetMethodDefinition((MethodDefinitionHandle)ctorHandle).GetDeclaringType();
            if (!delegateType.IsNil && Attrs(reader.GetTypeDefinition(delegateType).GetCustomAttributes()).Any(a => a.Contains("UnmanagedFunctionPointerAttribute", StringComparison.Ordinal)))
            {
                if (lastFunction.Kind != HandleKind.MethodDefinition) violations.Add("External reverse-call target");
                else
                {
                    var callback = reader.GetMethodDefinition((MethodDefinitionHandle)lastFunction);
                    var attributes = Attrs(callback.GetCustomAttributes());
                    if ((callback.Attributes & MethodAttributes.Static) == 0 || !attributes.Any(a => a.Contains("MonoPInvokeCallbackAttribute", StringComparison.Ordinal)))
                        violations.Add("Unannotated reverse-call target: " + Text(callback.Name));
                    else
                    {
                        var annotation = callback.GetCustomAttributes().Select(a => reader.GetCustomAttribute(a)).First(a =>
                            a.Constructor.Kind == HandleKind.MethodDefinition
                                ? Name(reader.GetMethodDefinition((MethodDefinitionHandle)a.Constructor).GetDeclaringType()).EndsWith(".MonoPInvokeCallbackAttribute", StringComparison.Ordinal)
                                : Name(reader.GetMemberReference((MemberReferenceHandle)a.Constructor).Parent).EndsWith(".MonoPInvokeCallbackAttribute", StringComparison.Ordinal));
                        var blob = reader.GetBlobReader(annotation.Value);
                        if (blob.ReadUInt16() != 1 || blob.ReadSerializedString()?.Split(',')[0] != Name(delegateType))
                            violations.Add("Wrong reverse-call delegate annotation: " + Text(callback.Name));
                    }
                }
            }
            lastFunction = default;
        }
    }
}
if (violations.Count != 0) throw new InvalidOperationException(string.Join("\n", violations));
File.WriteAllText(args[1], JsonSerializer.Serialize(surface, new JsonSerializerOptions { WriteIndented = true }) + "\n");

sealed class TypeNames(MetadataReader reader) : ISignatureTypeProvider<string, object?>
{
    public string Name(EntityHandle h) => h.Kind switch
    {
        HandleKind.TypeDefinition => GetTypeFromDefinition(reader, (TypeDefinitionHandle)h, 0),
        HandleKind.TypeReference => GetTypeFromReference(reader, (TypeReferenceHandle)h, 0),
        HandleKind.TypeSpecification => reader.GetTypeSpecification((TypeSpecificationHandle)h).DecodeSignature(this, null),
        _ => "?",
    };
    public string GetTypeFromDefinition(MetadataReader r, TypeDefinitionHandle h, byte raw)
    {
        var t = r.GetTypeDefinition(h); var parent = t.GetDeclaringType();
        return parent.IsNil ? Join(r.GetString(t.Namespace), r.GetString(t.Name)) : Name(parent) + "+" + r.GetString(t.Name);
    }
    public string GetTypeFromReference(MetadataReader r, TypeReferenceHandle h, byte raw)
    {
        var t = r.GetTypeReference(h);
        return t.ResolutionScope.Kind == HandleKind.TypeReference ? Name(t.ResolutionScope) + "+" + r.GetString(t.Name) : Join(r.GetString(t.Namespace), r.GetString(t.Name));
    }
    private static string Join(string ns, string name) => ns.Length == 0 ? name : ns + "." + name;
    public string GetTypeFromSpecification(MetadataReader r, object? context, TypeSpecificationHandle h, byte raw) => r.GetTypeSpecification(h).DecodeSignature(this, context);
    public string GetArrayType(string element, ArrayShape shape) => element + "[" + new string(',', shape.Rank - 1) + "]";
    public string GetByReferenceType(string element) => element + "@";
    public string GetFunctionPointerType(MethodSignature<string> signature) => "fnptr";
    public string GetGenericInstantiation(string type, ImmutableArray<string> args) => type + "{" + string.Join(",", args) + "}";
    public string GetGenericMethodParameter(object? context, int index) => "``" + index;
    public string GetGenericTypeParameter(object? context, int index) => "`" + index;
    public string GetModifiedType(string modifier, string type, bool required) => type + (required ? " modreq(" : " modopt(") + modifier + ")";
    public string GetPinnedType(string element) => element + " pinned";
    public string GetPointerType(string element) => element + "*";
    public string GetPrimitiveType(PrimitiveTypeCode code) => "System." + (code switch { PrimitiveTypeCode.Void => "Void", PrimitiveTypeCode.Boolean => "Boolean", PrimitiveTypeCode.Char => "Char", PrimitiveTypeCode.SByte => "SByte", PrimitiveTypeCode.Byte => "Byte", PrimitiveTypeCode.Int16 => "Int16", PrimitiveTypeCode.UInt16 => "UInt16", PrimitiveTypeCode.Int32 => "Int32", PrimitiveTypeCode.UInt32 => "UInt32", PrimitiveTypeCode.Int64 => "Int64", PrimitiveTypeCode.UInt64 => "UInt64", PrimitiveTypeCode.Single => "Single", PrimitiveTypeCode.Double => "Double", PrimitiveTypeCode.String => "String", PrimitiveTypeCode.Object => "Object", PrimitiveTypeCode.IntPtr => "IntPtr", PrimitiveTypeCode.UIntPtr => "UIntPtr", PrimitiveTypeCode.TypedReference => "TypedReference", _ => code.ToString() });
    public string GetSZArrayType(string element) => element + "[]";
}
