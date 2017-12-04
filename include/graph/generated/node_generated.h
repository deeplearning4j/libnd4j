// automatically generated by the FlatBuffers compiler, do not modify


#ifndef FLATBUFFERS_GENERATED_NODE_ND4J_GRAPH_H_
#define FLATBUFFERS_GENERATED_NODE_ND4J_GRAPH_H_

#include "flatbuffers/flatbuffers.h"

#include "array_generated.h"
#include "utils_generated.h"

namespace nd4j {
namespace graph {

struct FlatNode;

struct FlatNode FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  enum {
    VT_ID = 4,
    VT_NAME = 6,
    VT_OPTYPE = 8,
    VT_OPNUM = 10,
    VT_INPUT = 12,
    VT_INPUTPAIRED = 14,
    VT_DATATYPE = 16,
    VT_OUTPUT = 18,
    VT_EXTRAPARAMS = 20,
    VT_EXTRAINTEGER = 22,
    VT_DIMENSIONS = 24,
    VT_DEVICE = 26,
    VT_SCALAR = 28,
    VT_SCOPE_ID = 30,
    VT_SCOPE_NAME = 32
  };
  int32_t id() const {
    return GetField<int32_t>(VT_ID, 0);
  }
  const flatbuffers::String *name() const {
    return GetPointer<const flatbuffers::String *>(VT_NAME);
  }
  nd4j::graph::OpType opType() const {
    return static_cast<nd4j::graph::OpType>(GetField<int8_t>(VT_OPTYPE, 0));
  }
  int64_t opNum() const {
    return GetField<int64_t>(VT_OPNUM, 0);
  }
  const flatbuffers::Vector<int32_t> *input() const {
    return GetPointer<const flatbuffers::Vector<int32_t> *>(VT_INPUT);
  }
  const flatbuffers::Vector<flatbuffers::Offset<nd4j::graph::IntPair>> *inputPaired() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<nd4j::graph::IntPair>> *>(VT_INPUTPAIRED);
  }
  nd4j::graph::DataType dataType() const {
    return static_cast<nd4j::graph::DataType>(GetField<int8_t>(VT_DATATYPE, 0));
  }
  const flatbuffers::Vector<int32_t> *output() const {
    return GetPointer<const flatbuffers::Vector<int32_t> *>(VT_OUTPUT);
  }
  const flatbuffers::Vector<float> *extraParams() const {
    return GetPointer<const flatbuffers::Vector<float> *>(VT_EXTRAPARAMS);
  }
  const flatbuffers::Vector<int32_t> *extraInteger() const {
    return GetPointer<const flatbuffers::Vector<int32_t> *>(VT_EXTRAINTEGER);
  }
  const flatbuffers::Vector<int32_t> *dimensions() const {
    return GetPointer<const flatbuffers::Vector<int32_t> *>(VT_DIMENSIONS);
  }
  int32_t device() const {
    return GetField<int32_t>(VT_DEVICE, 0);
  }
  float scalar() const {
    return GetField<float>(VT_SCALAR, 0.0f);
  }
  int32_t scope_id() const {
    return GetField<int32_t>(VT_SCOPE_ID, 0);
  }
  const flatbuffers::String *scope_name() const {
    return GetPointer<const flatbuffers::String *>(VT_SCOPE_NAME);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<int32_t>(verifier, VT_ID) &&
           VerifyOffset(verifier, VT_NAME) &&
           verifier.Verify(name()) &&
           VerifyField<int8_t>(verifier, VT_OPTYPE) &&
           VerifyField<int64_t>(verifier, VT_OPNUM) &&
           VerifyOffset(verifier, VT_INPUT) &&
           verifier.Verify(input()) &&
           VerifyOffset(verifier, VT_INPUTPAIRED) &&
           verifier.Verify(inputPaired()) &&
           verifier.VerifyVectorOfTables(inputPaired()) &&
           VerifyField<int8_t>(verifier, VT_DATATYPE) &&
           VerifyOffset(verifier, VT_OUTPUT) &&
           verifier.Verify(output()) &&
           VerifyOffset(verifier, VT_EXTRAPARAMS) &&
           verifier.Verify(extraParams()) &&
           VerifyOffset(verifier, VT_EXTRAINTEGER) &&
           verifier.Verify(extraInteger()) &&
           VerifyOffset(verifier, VT_DIMENSIONS) &&
           verifier.Verify(dimensions()) &&
           VerifyField<int32_t>(verifier, VT_DEVICE) &&
           VerifyField<float>(verifier, VT_SCALAR) &&
           VerifyField<int32_t>(verifier, VT_SCOPE_ID) &&
           VerifyOffset(verifier, VT_SCOPE_NAME) &&
           verifier.Verify(scope_name()) &&
           verifier.EndTable();
  }
};

struct FlatNodeBuilder {
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_id(int32_t id) {
    fbb_.AddElement<int32_t>(FlatNode::VT_ID, id, 0);
  }
  void add_name(flatbuffers::Offset<flatbuffers::String> name) {
    fbb_.AddOffset(FlatNode::VT_NAME, name);
  }
  void add_opType(nd4j::graph::OpType opType) {
    fbb_.AddElement<int8_t>(FlatNode::VT_OPTYPE, static_cast<int8_t>(opType), 0);
  }
  void add_opNum(int64_t opNum) {
    fbb_.AddElement<int64_t>(FlatNode::VT_OPNUM, opNum, 0);
  }
  void add_input(flatbuffers::Offset<flatbuffers::Vector<int32_t>> input) {
    fbb_.AddOffset(FlatNode::VT_INPUT, input);
  }
  void add_inputPaired(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<nd4j::graph::IntPair>>> inputPaired) {
    fbb_.AddOffset(FlatNode::VT_INPUTPAIRED, inputPaired);
  }
  void add_dataType(nd4j::graph::DataType dataType) {
    fbb_.AddElement<int8_t>(FlatNode::VT_DATATYPE, static_cast<int8_t>(dataType), 0);
  }
  void add_output(flatbuffers::Offset<flatbuffers::Vector<int32_t>> output) {
    fbb_.AddOffset(FlatNode::VT_OUTPUT, output);
  }
  void add_extraParams(flatbuffers::Offset<flatbuffers::Vector<float>> extraParams) {
    fbb_.AddOffset(FlatNode::VT_EXTRAPARAMS, extraParams);
  }
  void add_extraInteger(flatbuffers::Offset<flatbuffers::Vector<int32_t>> extraInteger) {
    fbb_.AddOffset(FlatNode::VT_EXTRAINTEGER, extraInteger);
  }
  void add_dimensions(flatbuffers::Offset<flatbuffers::Vector<int32_t>> dimensions) {
    fbb_.AddOffset(FlatNode::VT_DIMENSIONS, dimensions);
  }
  void add_device(int32_t device) {
    fbb_.AddElement<int32_t>(FlatNode::VT_DEVICE, device, 0);
  }
  void add_scalar(float scalar) {
    fbb_.AddElement<float>(FlatNode::VT_SCALAR, scalar, 0.0f);
  }
  void add_scope_id(int32_t scope_id) {
    fbb_.AddElement<int32_t>(FlatNode::VT_SCOPE_ID, scope_id, 0);
  }
  void add_scope_name(flatbuffers::Offset<flatbuffers::String> scope_name) {
    fbb_.AddOffset(FlatNode::VT_SCOPE_NAME, scope_name);
  }
  FlatNodeBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  FlatNodeBuilder &operator=(const FlatNodeBuilder &);
  flatbuffers::Offset<FlatNode> Finish() {
    const auto end = fbb_.EndTable(start_, 15);
    auto o = flatbuffers::Offset<FlatNode>(end);
    return o;
  }
};

inline flatbuffers::Offset<FlatNode> CreateFlatNode(
    flatbuffers::FlatBufferBuilder &_fbb,
    int32_t id = 0,
    flatbuffers::Offset<flatbuffers::String> name = 0,
    nd4j::graph::OpType opType = nd4j::graph::OpType_TRANSFORM,
    int64_t opNum = 0,
    flatbuffers::Offset<flatbuffers::Vector<int32_t>> input = 0,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<nd4j::graph::IntPair>>> inputPaired = 0,
    nd4j::graph::DataType dataType = nd4j::graph::DataType_INHERIT,
    flatbuffers::Offset<flatbuffers::Vector<int32_t>> output = 0,
    flatbuffers::Offset<flatbuffers::Vector<float>> extraParams = 0,
    flatbuffers::Offset<flatbuffers::Vector<int32_t>> extraInteger = 0,
    flatbuffers::Offset<flatbuffers::Vector<int32_t>> dimensions = 0,
    int32_t device = 0,
    float scalar = 0.0f,
    int32_t scope_id = 0,
    flatbuffers::Offset<flatbuffers::String> scope_name = 0) {
  FlatNodeBuilder builder_(_fbb);
  builder_.add_opNum(opNum);
  builder_.add_scope_name(scope_name);
  builder_.add_scope_id(scope_id);
  builder_.add_scalar(scalar);
  builder_.add_device(device);
  builder_.add_dimensions(dimensions);
  builder_.add_extraInteger(extraInteger);
  builder_.add_extraParams(extraParams);
  builder_.add_output(output);
  builder_.add_inputPaired(inputPaired);
  builder_.add_input(input);
  builder_.add_name(name);
  builder_.add_id(id);
  builder_.add_dataType(dataType);
  builder_.add_opType(opType);
  return builder_.Finish();
}

inline flatbuffers::Offset<FlatNode> CreateFlatNodeDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    int32_t id = 0,
    const char *name = nullptr,
    nd4j::graph::OpType opType = nd4j::graph::OpType_TRANSFORM,
    int64_t opNum = 0,
    const std::vector<int32_t> *input = nullptr,
    const std::vector<flatbuffers::Offset<nd4j::graph::IntPair>> *inputPaired = nullptr,
    nd4j::graph::DataType dataType = nd4j::graph::DataType_INHERIT,
    const std::vector<int32_t> *output = nullptr,
    const std::vector<float> *extraParams = nullptr,
    const std::vector<int32_t> *extraInteger = nullptr,
    const std::vector<int32_t> *dimensions = nullptr,
    int32_t device = 0,
    float scalar = 0.0f,
    int32_t scope_id = 0,
    const char *scope_name = nullptr) {
  return nd4j::graph::CreateFlatNode(
      _fbb,
      id,
      name ? _fbb.CreateString(name) : 0,
      opType,
      opNum,
      input ? _fbb.CreateVector<int32_t>(*input) : 0,
      inputPaired ? _fbb.CreateVector<flatbuffers::Offset<nd4j::graph::IntPair>>(*inputPaired) : 0,
      dataType,
      output ? _fbb.CreateVector<int32_t>(*output) : 0,
      extraParams ? _fbb.CreateVector<float>(*extraParams) : 0,
      extraInteger ? _fbb.CreateVector<int32_t>(*extraInteger) : 0,
      dimensions ? _fbb.CreateVector<int32_t>(*dimensions) : 0,
      device,
      scalar,
      scope_id,
      scope_name ? _fbb.CreateString(scope_name) : 0);
}

inline const nd4j::graph::FlatNode *GetFlatNode(const void *buf) {
  return flatbuffers::GetRoot<nd4j::graph::FlatNode>(buf);
}

inline bool VerifyFlatNodeBuffer(
    flatbuffers::Verifier &verifier) {
  return verifier.VerifyBuffer<nd4j::graph::FlatNode>(nullptr);
}

inline void FinishFlatNodeBuffer(
    flatbuffers::FlatBufferBuilder &fbb,
    flatbuffers::Offset<nd4j::graph::FlatNode> root) {
  fbb.Finish(root);
}

}  // namespace graph
}  // namespace nd4j

#endif  // FLATBUFFERS_GENERATED_NODE_ND4J_GRAPH_H_
