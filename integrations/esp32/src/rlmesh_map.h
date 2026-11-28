// A robot's parts, described as values:
//
//   auto eyes = robot.add("eyes", rlmesh::Joints<2>{
//                                     .names = {"pan", "tilt"},
//                                     .hz = 50,
//                                     .command = rlmesh::Command::Position,
//                                     .safe = rlmesh::Safe::Hold,
//                                     .limits = {{-0.6, 0.6}, {-0.4, 0.4}},
//                                 });
//
// add() returns the part's handle: due() is its command for this control
// tick, publish() its sample.

#ifndef RLMESH_MAP_H_
#define RLMESH_MAP_H_

#include <WString.h>

#include <array>
#include <cstddef>
#include <optional>

namespace rlmesh {

class Robot;

// How a part is commanded; None publishes only.
enum class Command { None, Position, Velocity, Torque };

// What a part does when nothing drives it: keep its last target, or stop.
enum class Safe { Hold, Stop };

// A lower and an upper bound.
struct Range {
  float lo = 0;
  float hi = 0;
};

// One value from 0 (closed, off) to 1 (open, on), commanded by position.
struct Gripper {
  using Value = float;
  float hz;
  Safe safe = Safe::Hold;
};

// N joints by name. It publishes each joint's position; a commanded set
// takes one target per joint.
template <size_t N>
struct Joints {
  using Value = std::array<float, N>;
  std::array<const char*, N> names;
  float hz;
  Command command = Command::None;
  Safe safe = Safe::Hold;
  // One range per joint, enforced on the board whatever a policy sends.
  Range limits[N] = {};
};

// A part of your own kind ("light/led"): one field of `T`, a float or a
// std::array<float, N>. Naming a `command` makes it commandable.
template <typename T>
struct Custom {
  using Value = T;
  const char* kind;
  const char* field;
  float hz;
  const char* command = nullptr;
  Safe safe = Safe::Hold;
};

namespace detail {

// The map as the host reads it, written part by part.
class MapWriter {
 public:
  void open(const char* path, const char* kind) {
    json_ += json_.length() == 0 ? "{" : "},";
    quote(path);
    json_ += ":{\"kind\":";
    quote(kind);
  }
  void key(const char* name) {
    json_ += ',';
    quote(name);
    json_ += ':';
  }
  void number(const char* name, float value) {
    key(name);
    json_ += String(value, 4);
  }
  void text(const char* name, const char* value) {
    key(name);
    quote(value);
  }
  void raw(const char* text_) { json_ += text_; }
  void raw(float value) { json_ += String(value, 4); }
  void quote(const char* text_) {
    json_ += '"';
    for (const char* c = text_; *c != '\0'; ++c) {
      if (*c == '"' || *c == '\\') json_ += '\\';
      json_ += *c;
    }
    json_ += '"';
  }
  void commanded(Command command, Safe safe) {
    static const char* const kModes[] = {"", "position", "velocity", "torque"};
    if (command == Command::None) return;
    text("command", kModes[static_cast<int>(command)]);
    text("safe", safe == Safe::Hold ? "hold" : "stop");
  }
  const String& written() const { return json_; }
  String json() const { return json_.length() == 0 ? String("{}") : json_ + "}}"; }

 private:
  String json_;
};

inline void write(MapWriter& out, const char* path, const Gripper& part) {
  out.open(path, "gripper");
  out.number("hz", part.hz);
  out.commanded(Command::Position, part.safe);
}

template <size_t N>
void write(MapWriter& out, const char* path, const Joints<N>& part) {
  out.open(path, "joints");
  out.key("names");
  out.raw("[");
  for (size_t i = 0; i < N; ++i) {
    if (i != 0) out.raw(",");
    out.quote(part.names[i]);
  }
  out.raw("],\"fields\":[\"pos\"]");
  out.number("hz", part.hz);
  out.commanded(part.command, part.safe);
  bool limited = false;
  for (const Range& range : part.limits) limited = limited || range.lo != range.hi;
  if (limited && part.command != Command::None) {
    const char* const kFields[] = {"", "pos", "vel", "effort"};
    out.key("limits");
    out.raw("{");
    out.quote(kFields[static_cast<int>(part.command)]);
    out.raw(":[");
    for (size_t i = 0; i < N; ++i) {
      if (i != 0) out.raw(",");
      out.raw("[");
      out.raw(part.limits[i].lo);
      out.raw(",");
      out.raw(part.limits[i].hi);
      out.raw("]");
    }
    out.raw("]}");
  }
}

template <typename T>
void write(MapWriter& out, const char* path, const Custom<T>& part) {
  String schema = "{";
  out.open(path, part.kind);
  out.number("hz", part.hz);
  MapWriter field;
  field.quote(part.field);
  schema += field.written();
  schema += ":[\"float32\",[";
  schema += String(static_cast<unsigned>(sizeof(T) / sizeof(float)));
  schema += "]]}";
  out.key("fields");
  out.raw(schema.c_str());
  if (part.command != nullptr) {
    out.text("command", part.command);
    out.key("command_fields");
    out.raw(schema.c_str());
    out.text("safe", part.safe == Safe::Hold ? "hold" : "stop");
  }
}

}  // namespace detail

// One part of a robot: what Robot::add returns.
template <typename T>
class Part {
 public:
  // One control tick, at the part's rate: the command to apply now, if
  // there is one. With none, safe() says whether to run the safe behaviour.
  std::optional<T> due();
  // Whether the last due() asked for the part's safe behaviour: the
  // session is disarmed or e-stopped, or nothing has driven it for too long.
  bool safe() const { return safe_; }
  // Publishes the part's state, captured now. False before the robot runs.
  bool publish(const T& value);

 private:
  friend class Robot;
  Part(Robot& robot, const char* path) : robot_(&robot), path_(path) {}
  int index();
  Robot* robot_;
  const char* path_;
  int index_ = -1;
  bool safe_ = false;
};

}  // namespace rlmesh

#endif  // RLMESH_MAP_H_
