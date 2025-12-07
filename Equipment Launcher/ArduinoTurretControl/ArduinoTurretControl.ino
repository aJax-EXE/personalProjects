// #include <Servo.h>
// String inputString;

// Servo left_right;
// Servo up_down;

// void setup()
// {
//   left_right.attach(2);
//   up_down.attach(3);
//   Serial.begin(9600);
// }


// void loop()
// {
//   while(Serial.available())
//   {
//     inputString = Serial.readStringUntil('\r');
//     Serial.println(inputString);
//     int x_axis = inputString.substring(0, inputString.indexOf(',')).toInt();
//     int y_axis = inputString.substring(inputString.indexOf(',') + 1).toInt();

//     int y = map(y_axis, 0, 1080, 180, 0);
//     int x = map(x_axis, 0, 1920, 180, 0);

//     left_right.write(x);
//     up_down.write(y);
    
    

//     // Print the parsed values
//     Serial.print("First Integer: ");
//     Serial.println(x);
//     Serial.print("Second Integer: ");
//     Serial.println(y);
//   }
// }

#include <Servo.h>

Servo left_right;
Servo up_down;

String buffer = "";
unsigned long lastMoveTime = 0;

int targetX = 90;
int targetY = 90;

const int MOVE_INTERVAL = 20;   // ms between small servo steps (smooth movement)
const int STEP_AMOUNT = 2;      // how much servo changes per step

void setup() {
  left_right.attach(2);
  up_down.attach(3);

  Serial.begin(9600);
}

// Non-blocking serial read
void readSerial() {
  while (Serial.available()) {
    char c = Serial.read();

    if (c == '\r') {
      // parse full command
      int commaIndex = buffer.indexOf(',');

      if (commaIndex > 0) {
        int x_axis = buffer.substring(0, commaIndex).toInt();
        int y_axis = buffer.substring(commaIndex + 1).toInt();

        // Convert to servo angle
        targetY = map(y_axis, 0, 1080, 180, 0);
        targetX = map(x_axis, 0, 1920, 180, 0);

        // Constrain
        targetX = constrain(targetX, 0, 180);
        targetY = constrain(targetY, 0, 180);
      }

      buffer = "";  // clear after parsing
    }
    else {
      buffer += c;  // build message
    }
  }
}

// Smooth servo motion (prevents current spikes)
void updateServos() {
  unsigned long now = millis();
  if (now - lastMoveTime < MOVE_INTERVAL) return;
  lastMoveTime = now;

  int currentX = left_right.read();
  int currentY = up_down.read();

  // Smooth easing toward target
  if (currentX < targetX) currentX += STEP_AMOUNT;
  else if (currentX > targetX) currentX -= STEP_AMOUNT;

  if (currentY < targetY) currentY += STEP_AMOUNT;
  else if (currentY > targetY) currentY -= STEP_AMOUNT;

  currentX = constrain(currentX, 0, 180);
  currentY = constrain(currentY, 0, 180);

  left_right.write(currentX);
  up_down.write(currentY);
}

void loop() {
  readSerial();
  updateServos();
}
