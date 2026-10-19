using RLMesh;
using RLMesh.Spaces;
class Fixture
{
    static void Convert()
    {
        Value[] values = { true, (sbyte)1, (byte)1, (short)1, (ushort)1, 1, 1U, 1L, 1F, 1D, "a", new Metadata(), new Tensor(new byte[4], new long[] { 1 }, DType.Float32) };
        Bound[] bounds = { true, (sbyte)1, (byte)1, (short)1, (ushort)1, 1, 1U, 1L, 1UL, 1F, 1D };
        _ = new Box(0, 5, new long[] { 1 });
        _ = values; _ = bounds;
    }
}
