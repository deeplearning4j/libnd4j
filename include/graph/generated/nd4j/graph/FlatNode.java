// automatically generated by the FlatBuffers compiler, do not modify

package nd4j.graph;

import java.nio.*;
import java.lang.*;
import java.util.*;
import com.google.flatbuffers.*;

@SuppressWarnings("unused")
public final class FlatNode extends Table {
  public static FlatNode getRootAsFlatNode(ByteBuffer _bb) { return getRootAsFlatNode(_bb, new FlatNode()); }
  public static FlatNode getRootAsFlatNode(ByteBuffer _bb, FlatNode obj) { _bb.order(ByteOrder.LITTLE_ENDIAN); return (obj.__assign(_bb.getInt(_bb.position()) + _bb.position(), _bb)); }
  public void __init(int _i, ByteBuffer _bb) { bb_pos = _i; bb = _bb; }
  public FlatNode __assign(int _i, ByteBuffer _bb) { __init(_i, _bb); return this; }

  public int id() { int o = __offset(4); return o != 0 ? bb.getInt(o + bb_pos) : 0; }
  public String name() { int o = __offset(6); return o != 0 ? __string(o + bb_pos) : null; }
  public ByteBuffer nameAsByteBuffer() { return __vector_as_bytebuffer(6, 1); }
  public byte opType() { int o = __offset(8); return o != 0 ? bb.get(o + bb_pos) : 0; }
  public long opNum() { int o = __offset(10); return o != 0 ? bb.getLong(o + bb_pos) : 0L; }
  public FlatProperties properties(int j) { return properties(new FlatProperties(), j); }
  public FlatProperties properties(FlatProperties obj, int j) { int o = __offset(12); return o != 0 ? obj.__assign(__indirect(__vector(o) + j * 4), bb) : null; }
  public int propertiesLength() { int o = __offset(12); return o != 0 ? __vector_len(o) : 0; }
  public int input(int j) { int o = __offset(14); return o != 0 ? bb.getInt(__vector(o) + j * 4) : 0; }
  public int inputLength() { int o = __offset(14); return o != 0 ? __vector_len(o) : 0; }
  public ByteBuffer inputAsByteBuffer() { return __vector_as_bytebuffer(14, 4); }
  public IntPair inputPaired(int j) { return inputPaired(new IntPair(), j); }
  public IntPair inputPaired(IntPair obj, int j) { int o = __offset(16); return o != 0 ? obj.__assign(__indirect(__vector(o) + j * 4), bb) : null; }
  public int inputPairedLength() { int o = __offset(16); return o != 0 ? __vector_len(o) : 0; }
  public byte dataType() { int o = __offset(18); return o != 0 ? bb.get(o + bb_pos) : 0; }
  public int output(int j) { int o = __offset(20); return o != 0 ? bb.getInt(__vector(o) + j * 4) : 0; }
  public int outputLength() { int o = __offset(20); return o != 0 ? __vector_len(o) : 0; }
  public ByteBuffer outputAsByteBuffer() { return __vector_as_bytebuffer(20, 4); }
  public double extraParams(int j) { int o = __offset(22); return o != 0 ? bb.getDouble(__vector(o) + j * 8) : 0; }
  public int extraParamsLength() { int o = __offset(22); return o != 0 ? __vector_len(o) : 0; }
  public ByteBuffer extraParamsAsByteBuffer() { return __vector_as_bytebuffer(22, 8); }
  public long extraInteger(int j) { int o = __offset(24); return o != 0 ? bb.getLong(__vector(o) + j * 8) : 0; }
  public int extraIntegerLength() { int o = __offset(24); return o != 0 ? __vector_len(o) : 0; }
  public ByteBuffer extraIntegerAsByteBuffer() { return __vector_as_bytebuffer(24, 8); }
  public int dimensions(int j) { int o = __offset(26); return o != 0 ? bb.getInt(__vector(o) + j * 4) : 0; }
  public int dimensionsLength() { int o = __offset(26); return o != 0 ? __vector_len(o) : 0; }
  public ByteBuffer dimensionsAsByteBuffer() { return __vector_as_bytebuffer(26, 4); }
  public int device() { int o = __offset(28); return o != 0 ? bb.getInt(o + bb_pos) : 0; }
  public float scalar() { int o = __offset(30); return o != 0 ? bb.getFloat(o + bb_pos) : 0.0f; }
  public int scopeId() { int o = __offset(32); return o != 0 ? bb.getInt(o + bb_pos) : 0; }
  public String scopeName() { int o = __offset(34); return o != 0 ? __string(o + bb_pos) : null; }
  public ByteBuffer scopeNameAsByteBuffer() { return __vector_as_bytebuffer(34, 1); }

  public static int createFlatNode(FlatBufferBuilder builder,
      int id,
      int nameOffset,
      byte opType,
      long opNum,
      int propertiesOffset,
      int inputOffset,
      int inputPairedOffset,
      byte dataType,
      int outputOffset,
      int extraParamsOffset,
      int extraIntegerOffset,
      int dimensionsOffset,
      int device,
      float scalar,
      int scope_id,
      int scope_nameOffset) {
    builder.startObject(16);
    FlatNode.addOpNum(builder, opNum);
    FlatNode.addScopeName(builder, scope_nameOffset);
    FlatNode.addScopeId(builder, scope_id);
    FlatNode.addScalar(builder, scalar);
    FlatNode.addDevice(builder, device);
    FlatNode.addDimensions(builder, dimensionsOffset);
    FlatNode.addExtraInteger(builder, extraIntegerOffset);
    FlatNode.addExtraParams(builder, extraParamsOffset);
    FlatNode.addOutput(builder, outputOffset);
    FlatNode.addInputPaired(builder, inputPairedOffset);
    FlatNode.addInput(builder, inputOffset);
    FlatNode.addProperties(builder, propertiesOffset);
    FlatNode.addName(builder, nameOffset);
    FlatNode.addId(builder, id);
    FlatNode.addDataType(builder, dataType);
    FlatNode.addOpType(builder, opType);
    return FlatNode.endFlatNode(builder);
  }

  public static void startFlatNode(FlatBufferBuilder builder) { builder.startObject(16); }
  public static void addId(FlatBufferBuilder builder, int id) { builder.addInt(0, id, 0); }
  public static void addName(FlatBufferBuilder builder, int nameOffset) { builder.addOffset(1, nameOffset, 0); }
  public static void addOpType(FlatBufferBuilder builder, byte opType) { builder.addByte(2, opType, 0); }
  public static void addOpNum(FlatBufferBuilder builder, long opNum) { builder.addLong(3, opNum, 0L); }
  public static void addProperties(FlatBufferBuilder builder, int propertiesOffset) { builder.addOffset(4, propertiesOffset, 0); }
  public static int createPropertiesVector(FlatBufferBuilder builder, int[] data) { builder.startVector(4, data.length, 4); for (int i = data.length - 1; i >= 0; i--) builder.addOffset(data[i]); return builder.endVector(); }
  public static void startPropertiesVector(FlatBufferBuilder builder, int numElems) { builder.startVector(4, numElems, 4); }
  public static void addInput(FlatBufferBuilder builder, int inputOffset) { builder.addOffset(5, inputOffset, 0); }
  public static int createInputVector(FlatBufferBuilder builder, int[] data) { builder.startVector(4, data.length, 4); for (int i = data.length - 1; i >= 0; i--) builder.addInt(data[i]); return builder.endVector(); }
  public static void startInputVector(FlatBufferBuilder builder, int numElems) { builder.startVector(4, numElems, 4); }
  public static void addInputPaired(FlatBufferBuilder builder, int inputPairedOffset) { builder.addOffset(6, inputPairedOffset, 0); }
  public static int createInputPairedVector(FlatBufferBuilder builder, int[] data) { builder.startVector(4, data.length, 4); for (int i = data.length - 1; i >= 0; i--) builder.addOffset(data[i]); return builder.endVector(); }
  public static void startInputPairedVector(FlatBufferBuilder builder, int numElems) { builder.startVector(4, numElems, 4); }
  public static void addDataType(FlatBufferBuilder builder, byte dataType) { builder.addByte(7, dataType, 0); }
  public static void addOutput(FlatBufferBuilder builder, int outputOffset) { builder.addOffset(8, outputOffset, 0); }
  public static int createOutputVector(FlatBufferBuilder builder, int[] data) { builder.startVector(4, data.length, 4); for (int i = data.length - 1; i >= 0; i--) builder.addInt(data[i]); return builder.endVector(); }
  public static void startOutputVector(FlatBufferBuilder builder, int numElems) { builder.startVector(4, numElems, 4); }
  public static void addExtraParams(FlatBufferBuilder builder, int extraParamsOffset) { builder.addOffset(9, extraParamsOffset, 0); }
  public static int createExtraParamsVector(FlatBufferBuilder builder, double[] data) { builder.startVector(8, data.length, 8); for (int i = data.length - 1; i >= 0; i--) builder.addDouble(data[i]); return builder.endVector(); }
  public static void startExtraParamsVector(FlatBufferBuilder builder, int numElems) { builder.startVector(8, numElems, 8); }
  public static void addExtraInteger(FlatBufferBuilder builder, int extraIntegerOffset) { builder.addOffset(10, extraIntegerOffset, 0); }
  public static int createExtraIntegerVector(FlatBufferBuilder builder, long[] data) { builder.startVector(8, data.length, 8); for (int i = data.length - 1; i >= 0; i--) builder.addLong(data[i]); return builder.endVector(); }
  public static void startExtraIntegerVector(FlatBufferBuilder builder, int numElems) { builder.startVector(8, numElems, 8); }
  public static void addDimensions(FlatBufferBuilder builder, int dimensionsOffset) { builder.addOffset(11, dimensionsOffset, 0); }
  public static int createDimensionsVector(FlatBufferBuilder builder, int[] data) { builder.startVector(4, data.length, 4); for (int i = data.length - 1; i >= 0; i--) builder.addInt(data[i]); return builder.endVector(); }
  public static void startDimensionsVector(FlatBufferBuilder builder, int numElems) { builder.startVector(4, numElems, 4); }
  public static void addDevice(FlatBufferBuilder builder, int device) { builder.addInt(12, device, 0); }
  public static void addScalar(FlatBufferBuilder builder, float scalar) { builder.addFloat(13, scalar, 0.0f); }
  public static void addScopeId(FlatBufferBuilder builder, int scopeId) { builder.addInt(14, scopeId, 0); }
  public static void addScopeName(FlatBufferBuilder builder, int scopeNameOffset) { builder.addOffset(15, scopeNameOffset, 0); }
  public static int endFlatNode(FlatBufferBuilder builder) {
    int o = builder.endObject();
    return o;
  }
  public static void finishFlatNodeBuffer(FlatBufferBuilder builder, int offset) { builder.finish(offset); }
}

