// The echo env of python/rlmesh/tests/conformance/native_gate.py, for the C#
// gate program and the Unity host's.
using System;
using System.Collections.Generic;
using System.Linq;
using RLMesh;
using RLMesh.Spaces;

sealed class Unsupported(string why) : Exception(why);

sealed class EchoEnv : IEnv
{
    private readonly string row;
    private readonly Func<Value> first;
    private long t;

    public EchoEnv(string row)
    {
        this.row = row;
        (ObservationSpace, first) = Gate.Space(row);
    }
    public Space ObservationSpace { get; }
    public Space ActionSpace => ObservationSpace;
    public ResetResult Reset(long? seed, Metadata? options)
    {
        if (row == "error_reset") throw new ValueError("boom in reset");
        t = 0;
        var info = new Metadata();
        if (options != null) info["options"] = options;
        return new(first(), info);
    }
    public StepResult Step(Value? action)
    {
        if (row == "error_step") throw new RuntimeError("boom in step");
        bool done = ++t >= 3;
        var info = row == "info" ? Gate.RichInfo(t) : new Metadata { ["t"] = t };
        return new(action!, t, done, done && row == "both_done", info);
    }
}

static class Gate
{
    static int ItemSize(DType dtype) => dtype switch
    {
        DType.Bool or DType.Uint8 or DType.Int8 => 1,
        DType.Uint16 or DType.Int16 or DType.Float16 => 2,
        DType.Uint32 or DType.Int32 or DType.Float32 => 4,
        _ => 8,
    };

    // A space and a value in it.
    static (Space, Func<Value>) BoxOf(Bound low, Bound high, long[] shape, DType dtype = DType.Float32) =>
        (new Box(low, high, shape, new() { Dtype = dtype }),
         () => new Tensor(new byte[shape.Aggregate(1L, (n, d) => n * d) * ItemSize(dtype)], shape, dtype));
    static (Space, Func<Value>) DiscreteOf(long n, long start = 0) =>
        (new Discrete(n, new() { Start = start }), () => 0);
    static (Space, Func<Value>) DictOf(params (string Key, (Space Space, Func<Value> First) Child)[] children) =>
        (new Dict(children.Select(c => new KeyValuePair<string, Space>(c.Key, c.Child.Space))),
         () =>
         {
             var value = new Metadata();
             foreach (var (key, child) in children) value[key] = child.First();
             return value;
         });

    // The gate's SPACES, by name.
    internal static (Space, Func<Value>) Space(string row) => row switch
    {
        "box_f32_bounded" => BoxOf(-1.0, 2.0, new long[] { 3 }),
        "box_f64_inf" => BoxOf(double.NegativeInfinity, double.PositiveInfinity, new long[] { 2, 2 }, DType.Float64),
        "box_f16" => BoxOf(-1.0, 1.0, new long[] { 4 }, DType.Float16),
        "box_i8" => BoxOf(-5, 5, new long[] { 3 }, DType.Int8),
        "box_u8_image" => BoxOf(0, 255, new long[] { 6, 5, 3 }, DType.Uint8),
        "box_u16" => BoxOf(0, 60000, new long[] { 2 }, DType.Uint16),
        "box_i64" => BoxOf(-(1L << 40), 1L << 40, new long[] { 2 }, DType.Int64),
        "box_u64" => BoxOf(0, 1L << 40, new long[] { 2 }, DType.Uint64),
        "box_i16" => BoxOf(-300, 300, new long[] { 3 }, DType.Int16),
        "box_i32" => BoxOf(-(1 << 20), 1 << 20, new long[] { 3 }, DType.Int32),
        "box_u32" => BoxOf(0, 1 << 30, new long[] { 2 }, DType.Uint32),
        "box_bool" => BoxOf(0, 1, new long[] { 3 }, DType.Bool),
        "discrete" => DiscreteOf(4),
        "discrete_start" => DiscreteOf(5, -2),
        "dict_flat" => DictOf(("pos", BoxOf(-1, 1, new long[] { 2 })), ("gripper", DiscreteOf(2))),
        "dict_nested" => DictOf(
            ("robot", DictOf(("q", BoxOf(-1, 1, new long[] { 3 })), ("mode", DiscreteOf(2)))),
            ("scene", DictOf(("img", BoxOf(0, 255, new long[] { 4, 4, 3 }, DType.Uint8))))),
        "dict_keys_that_escape_alike" => DictOf(("Pos", DiscreteOf(3)), ("pos", DiscreteOf(5))),
        "dict_key_named_action" => DictOf(("action", BoxOf(-1, 1, new long[] { 1 }))),
        "box_mixed_bounds" => throw new Unsupported("C# v0.2 Box takes uniform bounds only"),
        "render" => throw new Unsupported("C# v0.2 env authoring has no render (IRender is design record, D-45)"),
        "scalar_box" => BoxOf(-1, 1, new long[0]),
        _ when row.StartsWith("multi_") || row.StartsWith("tuple") || row == "text" =>
            throw new Unsupported("C# v0.2 spaces are Box, Discrete and Dict (D-45)"),
        _ => BoxOf(-1, 1, new long[] { 3 }),
    };

    // The gate's rich_info(t).
    internal static Metadata RichInfo(long t) => new()
    {
        ["nested"] = new Metadata
        {
            ["a"] = new Metadata { ["b"] = t, ["c"] = Value.List(1, 2.5, "x") },
            ["flag"] = t % 2 == 0,
        },
        ["nan"] = double.NaN,
        ["inf"] = double.NegativeInfinity,
        ["bytes"] = Value.Bytes(new byte[] { 0x00, 0xff }),
        ["$dollar"] = 1,
        ["none"] = Value.None,
        ["third"] = 1.0 / 3,
        ["is_success"] = t >= 3,
    };

    /// <summary>Answers the gate's call `args` with `serve(env, address)` for
    /// `env ROW ADDRESS`; anything else is unsupported (D-45).</summary>
    internal static int Run(string[] args, Action<IEnv, string> serve)
    {
        try
        {
            if (args.Length != 3 || args[0] != "env")
                throw new Unsupported("the C# v0.2 surface serves envs only (D-45)");
            serve(new EchoEnv(args[1]), args[2]);
            return 0;
        }
        catch (Unsupported unsupported)
        {
            Console.WriteLine($"unsupported: {unsupported.Message}");
            return 0;
        }
        catch (Exception error)
        {
            Console.Error.WriteLine(error.Message);
            return 1;
        }
    }
}
