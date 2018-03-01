// automatically generated by the FlatBuffers compiler, do not modify


#ifndef FLATBUFFERS_GENERATED_CONFIG_ND4J_GRAPH_H_
#define FLATBUFFERS_GENERATED_CONFIG_ND4J_GRAPH_H_

#include "flatbuffers/flatbuffers.h"

namespace nd4j {
namespace graph {

struct FlatConfiguration;

enum ProfilingMode {
  ProfilingMode_NONE = 0,
  ProfilingMode_NAN_PANIC = 1,
  ProfilingMode_INF_PANIC = 2,
  ProfilingMode_ANY_PANIC = 3,
  ProfilingMode_MIN = ProfilingMode_NONE,
  ProfilingMode_MAX = ProfilingMode_ANY_PANIC
};

inline ProfilingMode (&EnumValuesProfilingMode())[4] {
  static ProfilingMode values[] = {
    ProfilingMode_NONE,
    ProfilingMode_NAN_PANIC,
    ProfilingMode_INF_PANIC,
    ProfilingMode_ANY_PANIC
  };
  return values;
}

inline const char **EnumNamesProfilingMode() {
  static const char *names[] = {
    "NONE",
    "NAN_PANIC",
    "INF_PANIC",
    "ANY_PANIC",
    nullptr
  };
  return names;
}

inline const char *EnumNameProfilingMode(ProfilingMode e) {
  const size_t index = static_cast<int>(e);
  return EnumNamesProfilingMode()[index];
}

enum ExecutionMode {
  ExecutionMode_SEQUENTIAL = 0,
  ExecutionMode_STRICT = 1,
  ExecutionMode_AUTO = 2,
  ExecutionMode_MIN = ExecutionMode_SEQUENTIAL,
  ExecutionMode_MAX = ExecutionMode_AUTO
};

inline ExecutionMode (&EnumValuesExecutionMode())[3] {
  static ExecutionMode values[] = {
    ExecutionMode_SEQUENTIAL,
    ExecutionMode_STRICT,
    ExecutionMode_AUTO
  };
  return values;
}

inline const char **EnumNamesExecutionMode() {
  static const char *names[] = {
    "SEQUENTIAL",
    "STRICT",
    "AUTO",
    nullptr
  };
  return names;
}

inline const char *EnumNameExecutionMode(ExecutionMode e) {
  const size_t index = static_cast<int>(e);
  return EnumNamesExecutionMode()[index];
}

enum OutputMode {
  OutputMode_IMPLICIT = 0,
  OutputMode_EXPLICIT = 1,
  OutputMode_EXPLICIT_AND_IMPLICIT = 2,
  OutputMode_VARIABLE_SPACE = 3,
  OutputMode_MIN = OutputMode_IMPLICIT,
  OutputMode_MAX = OutputMode_VARIABLE_SPACE
};

inline OutputMode (&EnumValuesOutputMode())[4] {
  static OutputMode values[] = {
    OutputMode_IMPLICIT,
    OutputMode_EXPLICIT,
    OutputMode_EXPLICIT_AND_IMPLICIT,
    OutputMode_VARIABLE_SPACE
  };
  return values;
}

inline const char **EnumNamesOutputMode() {
  static const char *names[] = {
    "IMPLICIT",
    "EXPLICIT",
    "EXPLICIT_AND_IMPLICIT",
    "VARIABLE_SPACE",
    nullptr
  };
  return names;
}

inline const char *EnumNameOutputMode(OutputMode e) {
  const size_t index = static_cast<int>(e);
  return EnumNamesOutputMode()[index];
}

enum Direction {
  Direction_FORWARD_ONLY = 0,
  Direction_FORWARD_AND_BACKWARD = 1,
  Direction_BACKWARD_ONLY = 2,
  Direction_MIN = Direction_FORWARD_ONLY,
  Direction_MAX = Direction_BACKWARD_ONLY
};

inline Direction (&EnumValuesDirection())[3] {
  static Direction values[] = {
    Direction_FORWARD_ONLY,
    Direction_FORWARD_AND_BACKWARD,
    Direction_BACKWARD_ONLY
  };
  return values;
}

inline const char **EnumNamesDirection() {
  static const char *names[] = {
    "FORWARD_ONLY",
    "FORWARD_AND_BACKWARD",
    "BACKWARD_ONLY",
    nullptr
  };
  return names;
}

inline const char *EnumNameDirection(Direction e) {
  const size_t index = static_cast<int>(e);
  return EnumNamesDirection()[index];
}

struct FlatConfiguration FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  enum {
    VT_ID = 4,
    VT_EXECUTIONMODE = 6,
    VT_PROFILINGMODE = 8,
    VT_OUTPUTMODE = 10,
    VT_TIMESTATS = 12,
    VT_FOOTPRINTFORWARD = 14,
    VT_FOOTPRINTBACKWARD = 16,
    VT_DIRECTION = 18
  };
  int64_t id() const {
    return GetField<int64_t>(VT_ID, 0);
  }
  ExecutionMode executionMode() const {
    return static_cast<ExecutionMode>(GetField<int8_t>(VT_EXECUTIONMODE, 0));
  }
  ProfilingMode profilingMode() const {
    return static_cast<ProfilingMode>(GetField<int8_t>(VT_PROFILINGMODE, 0));
  }
  OutputMode outputMode() const {
    return static_cast<OutputMode>(GetField<int8_t>(VT_OUTPUTMODE, 0));
  }
  bool timestats() const {
    return GetField<uint8_t>(VT_TIMESTATS, 0) != 0;
  }
  int64_t footprintForward() const {
    return GetField<int64_t>(VT_FOOTPRINTFORWARD, 0);
  }
  int64_t footprintBackward() const {
    return GetField<int64_t>(VT_FOOTPRINTBACKWARD, 0);
  }
  Direction direction() const {
    return static_cast<Direction>(GetField<int8_t>(VT_DIRECTION, 0));
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<int64_t>(verifier, VT_ID) &&
           VerifyField<int8_t>(verifier, VT_EXECUTIONMODE) &&
           VerifyField<int8_t>(verifier, VT_PROFILINGMODE) &&
           VerifyField<int8_t>(verifier, VT_OUTPUTMODE) &&
           VerifyField<uint8_t>(verifier, VT_TIMESTATS) &&
           VerifyField<int64_t>(verifier, VT_FOOTPRINTFORWARD) &&
           VerifyField<int64_t>(verifier, VT_FOOTPRINTBACKWARD) &&
           VerifyField<int8_t>(verifier, VT_DIRECTION) &&
           verifier.EndTable();
  }
};

struct FlatConfigurationBuilder {
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_id(int64_t id) {
    fbb_.AddElement<int64_t>(FlatConfiguration::VT_ID, id, 0);
  }
  void add_executionMode(ExecutionMode executionMode) {
    fbb_.AddElement<int8_t>(FlatConfiguration::VT_EXECUTIONMODE, static_cast<int8_t>(executionMode), 0);
  }
  void add_profilingMode(ProfilingMode profilingMode) {
    fbb_.AddElement<int8_t>(FlatConfiguration::VT_PROFILINGMODE, static_cast<int8_t>(profilingMode), 0);
  }
  void add_outputMode(OutputMode outputMode) {
    fbb_.AddElement<int8_t>(FlatConfiguration::VT_OUTPUTMODE, static_cast<int8_t>(outputMode), 0);
  }
  void add_timestats(bool timestats) {
    fbb_.AddElement<uint8_t>(FlatConfiguration::VT_TIMESTATS, static_cast<uint8_t>(timestats), 0);
  }
  void add_footprintForward(int64_t footprintForward) {
    fbb_.AddElement<int64_t>(FlatConfiguration::VT_FOOTPRINTFORWARD, footprintForward, 0);
  }
  void add_footprintBackward(int64_t footprintBackward) {
    fbb_.AddElement<int64_t>(FlatConfiguration::VT_FOOTPRINTBACKWARD, footprintBackward, 0);
  }
  void add_direction(Direction direction) {
    fbb_.AddElement<int8_t>(FlatConfiguration::VT_DIRECTION, static_cast<int8_t>(direction), 0);
  }
  explicit FlatConfigurationBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  FlatConfigurationBuilder &operator=(const FlatConfigurationBuilder &);
  flatbuffers::Offset<FlatConfiguration> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<FlatConfiguration>(end);
    return o;
  }
};

inline flatbuffers::Offset<FlatConfiguration> CreateFlatConfiguration(
    flatbuffers::FlatBufferBuilder &_fbb,
    int64_t id = 0,
    ExecutionMode executionMode = ExecutionMode_SEQUENTIAL,
    ProfilingMode profilingMode = ProfilingMode_NONE,
    OutputMode outputMode = OutputMode_IMPLICIT,
    bool timestats = false,
    int64_t footprintForward = 0,
    int64_t footprintBackward = 0,
    Direction direction = Direction_FORWARD_ONLY) {
  FlatConfigurationBuilder builder_(_fbb);
  builder_.add_footprintBackward(footprintBackward);
  builder_.add_footprintForward(footprintForward);
  builder_.add_id(id);
  builder_.add_direction(direction);
  builder_.add_timestats(timestats);
  builder_.add_outputMode(outputMode);
  builder_.add_profilingMode(profilingMode);
  builder_.add_executionMode(executionMode);
  return builder_.Finish();
}

inline const nd4j::graph::FlatConfiguration *GetFlatConfiguration(const void *buf) {
  return flatbuffers::GetRoot<nd4j::graph::FlatConfiguration>(buf);
}

inline bool VerifyFlatConfigurationBuffer(
    flatbuffers::Verifier &verifier) {
  return verifier.VerifyBuffer<nd4j::graph::FlatConfiguration>(nullptr);
}

inline void FinishFlatConfigurationBuffer(
    flatbuffers::FlatBufferBuilder &fbb,
    flatbuffers::Offset<nd4j::graph::FlatConfiguration> root) {
  fbb.Finish(root);
}

}  // namespace graph
}  // namespace nd4j

#endif  // FLATBUFFERS_GENERATED_CONFIG_ND4J_GRAPH_H_
