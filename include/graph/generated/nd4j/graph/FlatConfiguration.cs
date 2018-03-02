// <auto-generated>
//  automatically generated by the FlatBuffers compiler, do not modify
// </auto-generated>

namespace nd4j.graph
{

using global::System;
using global::FlatBuffers;

public struct FlatConfiguration : IFlatbufferObject
{
  private Table __p;
  public ByteBuffer ByteBuffer { get { return __p.bb; } }
  public static FlatConfiguration GetRootAsFlatConfiguration(ByteBuffer _bb) { return GetRootAsFlatConfiguration(_bb, new FlatConfiguration()); }
  public static FlatConfiguration GetRootAsFlatConfiguration(ByteBuffer _bb, FlatConfiguration obj) { return (obj.__assign(_bb.GetInt(_bb.Position) + _bb.Position, _bb)); }
  public void __init(int _i, ByteBuffer _bb) { __p.bb_pos = _i; __p.bb = _bb; }
  public FlatConfiguration __assign(int _i, ByteBuffer _bb) { __init(_i, _bb); return this; }

  public long Id { get { int o = __p.__offset(4); return o != 0 ? __p.bb.GetLong(o + __p.bb_pos) : (long)0; } }
  public ExecutionMode ExecutionMode { get { int o = __p.__offset(6); return o != 0 ? (ExecutionMode)__p.bb.GetSbyte(o + __p.bb_pos) : ExecutionMode.SEQUENTIAL; } }
  public ProfilingMode ProfilingMode { get { int o = __p.__offset(8); return o != 0 ? (ProfilingMode)__p.bb.GetSbyte(o + __p.bb_pos) : ProfilingMode.NONE; } }
  public OutputMode OutputMode { get { int o = __p.__offset(10); return o != 0 ? (OutputMode)__p.bb.GetSbyte(o + __p.bb_pos) : OutputMode.IMPLICIT; } }
  public bool Timestats { get { int o = __p.__offset(12); return o != 0 ? 0!=__p.bb.Get(o + __p.bb_pos) : (bool)false; } }
  public long FootprintForward { get { int o = __p.__offset(14); return o != 0 ? __p.bb.GetLong(o + __p.bb_pos) : (long)0; } }
  public long FootprintBackward { get { int o = __p.__offset(16); return o != 0 ? __p.bb.GetLong(o + __p.bb_pos) : (long)0; } }
  public Direction Direction { get { int o = __p.__offset(18); return o != 0 ? (Direction)__p.bb.GetSbyte(o + __p.bb_pos) : Direction.FORWARD_ONLY; } }

  public static Offset<FlatConfiguration> CreateFlatConfiguration(FlatBufferBuilder builder,
      long id = 0,
      ExecutionMode executionMode = ExecutionMode.SEQUENTIAL,
      ProfilingMode profilingMode = ProfilingMode.NONE,
      OutputMode outputMode = OutputMode.IMPLICIT,
      bool timestats = false,
      long footprintForward = 0,
      long footprintBackward = 0,
      Direction direction = Direction.FORWARD_ONLY) {
    builder.StartObject(8);
    FlatConfiguration.AddFootprintBackward(builder, footprintBackward);
    FlatConfiguration.AddFootprintForward(builder, footprintForward);
    FlatConfiguration.AddId(builder, id);
    FlatConfiguration.AddDirection(builder, direction);
    FlatConfiguration.AddTimestats(builder, timestats);
    FlatConfiguration.AddOutputMode(builder, outputMode);
    FlatConfiguration.AddProfilingMode(builder, profilingMode);
    FlatConfiguration.AddExecutionMode(builder, executionMode);
    return FlatConfiguration.EndFlatConfiguration(builder);
  }

  public static void StartFlatConfiguration(FlatBufferBuilder builder) { builder.StartObject(8); }
  public static void AddId(FlatBufferBuilder builder, long id) { builder.AddLong(0, id, 0); }
  public static void AddExecutionMode(FlatBufferBuilder builder, ExecutionMode executionMode) { builder.AddSbyte(1, (sbyte)executionMode, 0); }
  public static void AddProfilingMode(FlatBufferBuilder builder, ProfilingMode profilingMode) { builder.AddSbyte(2, (sbyte)profilingMode, 0); }
  public static void AddOutputMode(FlatBufferBuilder builder, OutputMode outputMode) { builder.AddSbyte(3, (sbyte)outputMode, 0); }
  public static void AddTimestats(FlatBufferBuilder builder, bool timestats) { builder.AddBool(4, timestats, false); }
  public static void AddFootprintForward(FlatBufferBuilder builder, long footprintForward) { builder.AddLong(5, footprintForward, 0); }
  public static void AddFootprintBackward(FlatBufferBuilder builder, long footprintBackward) { builder.AddLong(6, footprintBackward, 0); }
  public static void AddDirection(FlatBufferBuilder builder, Direction direction) { builder.AddSbyte(7, (sbyte)direction, 0); }
  public static Offset<FlatConfiguration> EndFlatConfiguration(FlatBufferBuilder builder) {
    int o = builder.EndObject();
    return new Offset<FlatConfiguration>(o);
  }
  public static void FinishFlatConfigurationBuffer(FlatBufferBuilder builder, Offset<FlatConfiguration> offset) { builder.Finish(offset.Value); }
};


}
