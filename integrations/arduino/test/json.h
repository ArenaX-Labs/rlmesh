// A small JSON reader for the conformance vectors: objects, arrays,
// strings, non-negative integers, booleans and null. Host only.

#ifndef RLMESH_TETHER_TEST_JSON_H_
#define RLMESH_TETHER_TEST_JSON_H_

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <string>
#include <utility>
#include <vector>

namespace json {

struct Value {
  enum Kind { kNull, kBool, kInt, kString, kArray, kObject } kind = kNull;
  bool boolean = false;
  uint64_t integer = 0;
  std::string string;
  std::vector<Value> items;
  std::vector<std::pair<std::string, Value>> members;

  // The member `key`, or null.
  const Value& operator[](const char* key) const {
    static const Value null;
    for (const auto& m : members) {
      if (m.first == key) return m.second;
    }
    return null;
  }
  bool has(const char* key) const { return (*this)[key].kind != kNull; }
};

class Parser {
 public:
  explicit Parser(const std::string& text) : s_(text) {}

  Value parse() {
    Value v = value();
    space();
    if (i_ != s_.size()) fail("trailing data");
    return v;
  }

 private:
  [[noreturn]] void fail(const char* what) {
    fprintf(stderr, "json: %s at byte %zu\n", what, i_);
    exit(2);
  }

  void space() {
    while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\n' || s_[i_] == '\r' || s_[i_] == '\t'))
      ++i_;
  }

  bool eat(char c) {
    space();
    if (i_ < s_.size() && s_[i_] == c) {
      ++i_;
      return true;
    }
    return false;
  }

  void expect(char c) {
    if (!eat(c)) fail("unexpected character");
  }

  bool word(const char* w) {
    size_t n = 0;
    while (w[n]) ++n;
    if (s_.compare(i_, n, w) != 0) return false;
    i_ += n;
    return true;
  }

  std::string str() {
    expect('"');
    std::string out;
    while (i_ < s_.size() && s_[i_] != '"') {
      char c = s_[i_++];
      if (c == '\\') {
        if (i_ >= s_.size()) fail("bad escape");
        char e = s_[i_++];
        switch (e) {
          case 'n':
            out += '\n';
            break;
          case 't':
            out += '\t';
            break;
          case 'u':
            // Only labels use escapes; keep a placeholder.
            i_ += 4;
            out += '?';
            break;
          default:
            out += e;
        }
      } else {
        out += c;
      }
    }
    expect('"');
    return out;
  }

  Value value() {
    space();
    Value v;
    if (i_ >= s_.size()) fail("unexpected end");
    char c = s_[i_];
    if (c == '{') {
      ++i_;
      v.kind = Value::kObject;
      if (eat('}')) return v;
      do {
        std::string key = str();
        expect(':');
        Value item = value();
        // A repeated key keeps its last value, as serde_json does.
        bool replaced = false;
        for (auto& m : v.members) {
          if (m.first == key) {
            m.second = item;
            replaced = true;
          }
        }
        if (!replaced) v.members.emplace_back(key, item);
      } while (eat(','));
      expect('}');
    } else if (c == '[') {
      ++i_;
      v.kind = Value::kArray;
      if (eat(']')) return v;
      do v.items.push_back(value());
      while (eat(','));
      expect(']');
    } else if (c == '"') {
      v.kind = Value::kString;
      v.string = str();
    } else if (c >= '0' && c <= '9') {
      v.kind = Value::kInt;
      while (i_ < s_.size() && s_[i_] >= '0' && s_[i_] <= '9')
        v.integer = v.integer * 10 + static_cast<uint64_t>(s_[i_++] - '0');
    } else if (word("true")) {
      v.kind = Value::kBool;
      v.boolean = true;
    } else if (word("false")) {
      v.kind = Value::kBool;
    } else if (!word("null")) {
      fail("unexpected value");
    }
    return v;
  }

  const std::string& s_;
  size_t i_ = 0;
};

}  // namespace json

#endif  // RLMESH_TETHER_TEST_JSON_H_
