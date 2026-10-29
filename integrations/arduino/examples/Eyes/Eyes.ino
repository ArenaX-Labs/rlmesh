// Two hobby servos as a pair of eyes, driven by an RLMesh policy through the
// computer this board is plugged into (`rlmesh tether /dev/ttyACM0`).

#include <RLMeshTether.h>
#include <Servo.h>

#include "eyes_map.h"

Servo pan, tilt;
RLMeshTether<EyesMap> tether(Serial);
float last[2] = {0, 0};

int toDeg(float rad) { return 90 + int(rad * 57.2958f); }

void onEyes(const EyesCmd& cmd, bool hold) {
  (void)hold;  // a hold repeats the last target
  pan.write(toDeg(cmd.pos[0]));
  tilt.write(toDeg(cmd.pos[1]));
  last[0] = cmd.pos[0];
  last[1] = cmd.pos[1];
}

void onSafe(Part part, SafeBehaviour how, bool estop) {
  // EYES_SAFE is `hold`: keep the last angle. A `stop` part would detach()
  // its servos here.
  (void)part;
  (void)how;
  (void)estop;
}

void setup() {
  pan.attach(9);
  tilt.attach(10);
  Serial.begin(115200);
  tether.onCommand(EYES, onEyes);
  tether.begin(onSafe, /*watchdogMs=*/250);  // last: the watchdog arms on the first poll()
}

void loop() {
  tether.poll();
  // No position sensors here: report the last target as the state.
  tether.every(20, [] { tether.publish(EYES, EyesState{{last[0], last[1]}}); });
  // Use tether.delay(ms) instead of delay(ms), and tether.print() instead of Serial.print().
}
