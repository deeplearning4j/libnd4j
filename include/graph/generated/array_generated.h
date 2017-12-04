// automatically generated by the FlatBuffers compiler, do not modify


#ifndef FLATBUFFERS_GENERATED_ARRAY_ND4J_GRAPH_H_
#define FLATBUFFERS_GENERATED_ARRAY_ND4J_GRAPH_H_

#include "flatbuffers/flatbuffers.h"

namespace nd4j {
namespace graph {

struct FlatArray;

enum ByteOrder {
  ByteOrder_LE = 0,
  ByteOrder_BE = 1,
  ByteOrder_MIN = ByteOrder_LE,
  ByteOrder_MAX = ByteOrder_BE
};

inline ByteOrder (&EnumValuesByteOrder())[2] {
  static ByteOrder values[] = {
    ByteOrder_LE,
    ByteOrder_BE
  };
  return values;
}

inline const char **EnumNamesByteOrder() {
  static const char *names[] = {
    "LE",
    "BE",
    nullptr
  };
  return names;
}

inline const char *EnumNameByteOrder(ByteOrder e) {
  const size_t index = static_cast<int>(e);
  return EnumNamesByteOrder()[index];
}

enum DataType {
  DataType_INHERIT = 0,
  DataType_BOOL = 1,
  DataType_FLOAT8 = 2,
  DataType_HALF = 3,
  DataType_HALF2 = 4,
  DataType_FLOAT = 5,
  DataType_DOUBLE = 6,
  DataType_INT8 = 7,
  DataType_INT16 = 8,
  DataType_INT32 = 9,
  DataType_INT64 = 10,
  DataType_UINT8 = 11,
  DataType_UINT16 = 12,
  DataType_UINT32 = 13,
  DataType_UINT64 = 14,
  DataType_QINT8 = 15,
  DataType_QINT16 = 16,
  DataType_MIN = DataType_INHERIT,
  DataType_MAX = DataType_QINT16
};

inline DataType (&EnumValuesDataType())[17] {
  static DataType values[] = {
    DataType_INHERIT,
    DataType_BOOL,
    DataType_FLOAT8,
    DataType_HALF,
    DataType_HALF2,
    DataType_FLOAT,
    DataType_DOUBLE,
    DataType_INT8,
    DataType_INT16,
    DataType_INT32,
    DataType_INT64,
    DataType_UINT8,
    DataType_UINT16,
    DataType_UINT32,
    DataType_UINT64,
    DataType_QINT8,
    DataType_QINT16
  };
  return values;
}

inline const char **EnumNamesDataType() {
  static const char *names[] = {
    "INHERIT",
    "BOOL",
    "FLOAT8",
    "HALF",
    "HALF2",
    "FLOAT",
    "DOUBLE",
    "INT8",
    "INT16",
    "INT32",
    "INT64",
    "UINT8",
    "UINT16",
    "UINT32",
    "UINT64",
    "QINT8",
    "QINT16",
    nullptr
  };
  return names;
}

inline const char *EnumNameDataType(DataType e) {
  const size_t index = static_cast<int>(e);
  return EnumNamesDataType()[index];
}

struct FlatArray FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  enum {
    VT_SHAPE = 4,
    VT_BUFFER = 6,
    VT_DTYPE = 8,
    VT_BYTEORDER = 10
  };
  const flatbuffers::Vector<int32_t> *shape() const {
    return GetPointer<const flatbuffers::Vector<int32_t> *>(VT_SHAPE);
  }
  const flatbuffers::Vector<int8_t> *buffer() const {
    return GetPointer<const flatbuffers::Vector<int8_t> *>(VT_BUFFER);
  }
  DataType dtype() const {
    return static_cast<DataType>(GetField<int8_t>(VT_DTYPE, 0));
  }
  ByteOrder byteOrder() const {
    return static_cast<ByteOrder>(GetField<int8_t>(VT_BYTEORDER, 0));
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_SHAPE) &&
           verifier.Verify(shape()) &&
           VerifyOffset(verifier, VT_BUFFER) &&
           verifier.Verify(buffer()) &&
           VerifyField<int8_t>(verifier, VT_DTYPE) &&
           VerifyField<int8_t>(verifier, VT_BYTEORDER) &&
           verifier.EndTable();
  }
};

struct FlatArrayBuilder {
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_shape(flatbuffers::Offset<flatbuffers::Vector<int32_t>> shape) {
    fbb_.AddOffset(FlatArray::VT_SHAPE, shape);
  }
  void add_buffer(flatbuffers::Offset<flatbuffers::Vector<int8_t>> buffer) {
    fbb_.AddOffset(FlatArray::VT_BUFFER, buffer);
  }
  void add_dtype(DataType dtype) {
    fbb_.AddElement<int8_t>(FlatArray::VT_DTYPE, static_cast<int8_t>(dtype), 0);
  }
  void add_byteOrder(ByteOrder byteOrder) {
    fbb_.AddElement<int8_t>(FlatArray::VT_BYTEORDER, static_cast<int8_t>(byteOrder), 0);
  }
  FlatArrayBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  FlatArrayBuilder &operator=(const FlatArrayBuilder &);
  flatbuffers::Offset<FlatArray> Finish() {
    const auto end = fbb_.EndTable(start_, 4);
    auto o = flatbuffers::Offset<FlatArray>(end);
    return o;
  }
};

inline flatbuffers::Offset<FlatArray> CreateFlatArray(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<flatbuffers::Vector<int32_t>> shape = 0,
    flatbuffers::Offset<flatbuffers::Vector<int8_t>> buffer = 0,
    DataType dtype = DataType_INHERIT,
    ByteOrder byteOrder = ByteOrder_LE) {
  FlatArrayBuilder builder_(_fbb);
  builder_.add_buffer(buffer);
  builder_.add_shape(shape);
  builder_.add_byteOrder(byteOrder);
  builder_.add_dtype(dtype);
  return builder_.Finish();
}

inline flatbuffers::Offset<FlatArray> CreateFlatArrayDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    const std::vector<int32_t> *shape = nullptr,
    const std::vector<int8_t> *buffer = nullptr,
    DataType dtype = DataType_INHERIT,
    ByteOrder byteOrder = ByteOrder_LE) {
  return nd4j::graph::CreateFlatArray(
      _fbb,
      shape ? _fbb.CreateVector<int32_t>(*shape) : 0,
      buffer ? _fbb.CreateVector<int8_t>(*buffer) : 0,
      dtype,
      byteOrder);
}

inline const nd4j::graph::FlatArray *GetFlatArray(const void *buf) {
  return flatbuffers::GetRoot<nd4j::graph::FlatArray>(buf);
}

inline bool VerifyFlatArrayBuffer(
    flatbuffers::Verifier &verifier) {
  return verifier.VerifyBuffer<nd4j::graph::FlatArray>(nullptr);
}

inline void FinishFlatArrayBuffer(
    flatbuffers::FlatBufferBuilder &fbb,
    flatbuffers::Offset<nd4j::graph::FlatArray> root) {
  fbb.Finish(root);
}

}  // namespace graph
}  // namespace nd4j

#endif  // FLATBUFFERS_GENERATED_ARRAY_ND4J_GRAPH_H_
