// automatically generated by the FlatBuffers compiler, do not modify


#ifndef FLATBUFFERS_GENERATED_UTILS_ND4J_GRAPH_H_
#define FLATBUFFERS_GENERATED_UTILS_ND4J_GRAPH_H_

#include "flatbuffers/flatbuffers.h"

namespace nd4j {
namespace graph {

struct LongPair;

struct IntPair;

struct IntTriple;

enum OpType {
  OpType_TRANSFORM = 0,
  OpType_ACCUMULATION = 1,
  OpType_INDEX_ACCUMULATION = 2,
  OpType_SCALAR = 3,
  OpType_BROADCAST = 4,
  OpType_SUMMARYSTATS = 7,
  OpType_SHAPE = 8,
  OpType_AGGREGATION = 9,
  OpType_CUSTOM = 10,
  OpType_GRAPH = 11,
  OpType_VARIABLE = 30,
  OpType_BOOLEAN = 40,
  OpType_LOGIC = 119,
  OpType_MIN = OpType_TRANSFORM,
  OpType_MAX = OpType_LOGIC
};

inline OpType (&EnumValuesOpType())[13] {
  static OpType values[] = {
    OpType_TRANSFORM,
    OpType_ACCUMULATION,
    OpType_INDEX_ACCUMULATION,
    OpType_SCALAR,
    OpType_BROADCAST,
    OpType_SUMMARYSTATS,
    OpType_SHAPE,
    OpType_AGGREGATION,
    OpType_CUSTOM,
    OpType_GRAPH,
    OpType_VARIABLE,
    OpType_BOOLEAN,
    OpType_LOGIC
  };
  return values;
}

enum InputType {
  InputType_UNDEFINED = 0,
  InputType_NUMERIC = 1,
  InputType_STRINGULAR = 2,
  InputType_NUMERIC_SET = 3,
  InputType_STRINGULAR_SET = 4,
  InputType_MIN = InputType_UNDEFINED,
  InputType_MAX = InputType_STRINGULAR_SET
};

inline InputType (&EnumValuesInputType())[5] {
  static InputType values[] = {
    InputType_UNDEFINED,
    InputType_NUMERIC,
    InputType_STRINGULAR,
    InputType_NUMERIC_SET,
    InputType_STRINGULAR_SET
  };
  return values;
}

inline const char **EnumNamesInputType() {
  static const char *names[] = {
    "UNDEFINED",
    "NUMERIC",
    "STRINGULAR",
    "NUMERIC_SET",
    "STRINGULAR_SET",
    nullptr
  };
  return names;
}

inline const char *EnumNameInputType(InputType e) {
  const size_t index = static_cast<int>(e);
  return EnumNamesInputType()[index];
}

enum OpClass {
  OpClass_TRANSFORM = 0,
  OpClass_REDUCTION = 1,
  OpClass_MULTIPLICATOR = 2,
  OpClass_GRAPH = 3,
  OpClass_CONDITIONAL = 4,
  OpClass_LOOP = 5,
  OpClass_MIN = OpClass_TRANSFORM,
  OpClass_MAX = OpClass_LOOP
};

inline OpClass (&EnumValuesOpClass())[6] {
  static OpClass values[] = {
    OpClass_TRANSFORM,
    OpClass_REDUCTION,
    OpClass_MULTIPLICATOR,
    OpClass_GRAPH,
    OpClass_CONDITIONAL,
    OpClass_LOOP
  };
  return values;
}

inline const char **EnumNamesOpClass() {
  static const char *names[] = {
    "TRANSFORM",
    "REDUCTION",
    "MULTIPLICATOR",
    "GRAPH",
    "CONDITIONAL",
    "LOOP",
    nullptr
  };
  return names;
}

inline const char *EnumNameOpClass(OpClass e) {
  const size_t index = static_cast<int>(e);
  return EnumNamesOpClass()[index];
}

struct LongPair FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  enum {
    VT_FIRST = 4,
    VT_SECOND = 6
  };
  int64_t first() const {
    return GetField<int64_t>(VT_FIRST, 0);
  }
  int64_t second() const {
    return GetField<int64_t>(VT_SECOND, 0);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<int64_t>(verifier, VT_FIRST) &&
           VerifyField<int64_t>(verifier, VT_SECOND) &&
           verifier.EndTable();
  }
};

struct LongPairBuilder {
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_first(int64_t first) {
    fbb_.AddElement<int64_t>(LongPair::VT_FIRST, first, 0);
  }
  void add_second(int64_t second) {
    fbb_.AddElement<int64_t>(LongPair::VT_SECOND, second, 0);
  }
  LongPairBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  LongPairBuilder &operator=(const LongPairBuilder &);
  flatbuffers::Offset<LongPair> Finish() {
    const auto end = fbb_.EndTable(start_, 2);
    auto o = flatbuffers::Offset<LongPair>(end);
    return o;
  }
};

inline flatbuffers::Offset<LongPair> CreateLongPair(
    flatbuffers::FlatBufferBuilder &_fbb,
    int64_t first = 0,
    int64_t second = 0) {
  LongPairBuilder builder_(_fbb);
  builder_.add_second(second);
  builder_.add_first(first);
  return builder_.Finish();
}

struct IntPair FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  enum {
    VT_FIRST = 4,
    VT_SECOND = 6
  };
  int32_t first() const {
    return GetField<int32_t>(VT_FIRST, 0);
  }
  int32_t second() const {
    return GetField<int32_t>(VT_SECOND, 0);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<int32_t>(verifier, VT_FIRST) &&
           VerifyField<int32_t>(verifier, VT_SECOND) &&
           verifier.EndTable();
  }
};

struct IntPairBuilder {
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_first(int32_t first) {
    fbb_.AddElement<int32_t>(IntPair::VT_FIRST, first, 0);
  }
  void add_second(int32_t second) {
    fbb_.AddElement<int32_t>(IntPair::VT_SECOND, second, 0);
  }
  IntPairBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  IntPairBuilder &operator=(const IntPairBuilder &);
  flatbuffers::Offset<IntPair> Finish() {
    const auto end = fbb_.EndTable(start_, 2);
    auto o = flatbuffers::Offset<IntPair>(end);
    return o;
  }
};

inline flatbuffers::Offset<IntPair> CreateIntPair(
    flatbuffers::FlatBufferBuilder &_fbb,
    int32_t first = 0,
    int32_t second = 0) {
  IntPairBuilder builder_(_fbb);
  builder_.add_second(second);
  builder_.add_first(first);
  return builder_.Finish();
}

struct IntTriple FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  enum {
    VT_FIRST = 4,
    VT_SECOND = 6,
    VT_THIRD = 8
  };
  int32_t first() const {
    return GetField<int32_t>(VT_FIRST, 0);
  }
  int32_t second() const {
    return GetField<int32_t>(VT_SECOND, 0);
  }
  int32_t third() const {
    return GetField<int32_t>(VT_THIRD, 0);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<int32_t>(verifier, VT_FIRST) &&
           VerifyField<int32_t>(verifier, VT_SECOND) &&
           VerifyField<int32_t>(verifier, VT_THIRD) &&
           verifier.EndTable();
  }
};

struct IntTripleBuilder {
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_first(int32_t first) {
    fbb_.AddElement<int32_t>(IntTriple::VT_FIRST, first, 0);
  }
  void add_second(int32_t second) {
    fbb_.AddElement<int32_t>(IntTriple::VT_SECOND, second, 0);
  }
  void add_third(int32_t third) {
    fbb_.AddElement<int32_t>(IntTriple::VT_THIRD, third, 0);
  }
  IntTripleBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  IntTripleBuilder &operator=(const IntTripleBuilder &);
  flatbuffers::Offset<IntTriple> Finish() {
    const auto end = fbb_.EndTable(start_, 3);
    auto o = flatbuffers::Offset<IntTriple>(end);
    return o;
  }
};

inline flatbuffers::Offset<IntTriple> CreateIntTriple(
    flatbuffers::FlatBufferBuilder &_fbb,
    int32_t first = 0,
    int32_t second = 0,
    int32_t third = 0) {
  IntTripleBuilder builder_(_fbb);
  builder_.add_third(third);
  builder_.add_second(second);
  builder_.add_first(first);
  return builder_.Finish();
}

}  // namespace graph
}  // namespace nd4j

#endif  // FLATBUFFERS_GENERATED_UTILS_ND4J_GRAPH_H_
