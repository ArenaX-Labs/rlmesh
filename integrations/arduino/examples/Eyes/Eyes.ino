#include <RLMeshTether.h>
#include <Servo.h>

#include "eyes_map.h"

Servo pan, tilt;
RLMeshTether<EyesMap> tether(Serial);
float last[2] = {0, 0};

int toDeg(float rad) { return 90 + int(rad * 57.2958f); }

void onEyes(const EyesCmd& cmd, bool hold) {
  (void)hold;
  pan.write(toDeg(cmd.pos[0]));
  tilt.write(toDeg(cmd.pos[1]));
  last[0] = cmd.pos[0];
  last[1] = cmd.pos[1];
}

void onSafe(Part part, SafeBehaviour how, bool estop) {
  (void)part;
  (void)how;
  (void)estop;
}

void setup() {
  pan.attach(9);
  tilt.attach(10);
  Serial.begin(115200);
  tether.onCommand(EYES, onEyes);
  tether.begin(onSafe, 250);
}

void loop() {
  tether.poll();
  tether.every(20, [] { tether.publish(EYES, EyesState{{last[0], last[1]}}); });
}
