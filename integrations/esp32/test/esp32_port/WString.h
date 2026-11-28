#ifndef WSTRING_H_
#define WSTRING_H_

#include <cstdio>
#include <string>

class String {
 public:
  String() = default;
  String(const char* text) : s_(text) {}
  String(unsigned value) : s_(std::to_string(value)) {}
  String(float value, unsigned char decimals) {
    char buf[48];
    snprintf(buf, sizeof buf, "%.*f", decimals, static_cast<double>(value));
    s_ = buf;
  }
  String& operator+=(const String& other) {
    s_ += other.s_;
    return *this;
  }
  String& operator+=(const char* text) {
    s_ += text;
    return *this;
  }
  String& operator+=(char c) {
    s_ += c;
    return *this;
  }
  String operator+(const char* text) const {
    String out = *this;
    out += text;
    return out;
  }
  unsigned length() const { return static_cast<unsigned>(s_.size()); }
  const char* c_str() const { return s_.c_str(); }

 private:
  std::string s_;
};

#endif
