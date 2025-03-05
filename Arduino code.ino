void setup() {
    // Start serial communication at a baud rate of 9600
    Serial.begin(9600);
  
    // Set up pins for LEDs (or motors)
    pinMode(13, OUTPUT);  // Red LED
    pinMode(12, OUTPUT);  // Blue LED
    pinMode(11, OUTPUT);  // Green LED
    pinMode(10, OUTPUT);  // Yellow LED
  }
  
  void loop() {
    // Check if there's incoming serial data
    if (Serial.available() > 0) {
      char receivedChar = Serial.read();  // Read the incoming byte
     // Serial.print(receivedChar);
      // Turn on the corresponding LED based on the received character
      if (receivedChar == 'R') {
        digitalWrite(13, HIGH);  // Turn on Red LED
        digitalWrite(12, LOW);   // Turn off other LEDs
        digitalWrite(11, LOW);
        digitalWrite(10, LOW);
        delay(2000);
      } else if (receivedChar == 'B') {
        digitalWrite(12, HIGH);  // Turn on Blue LED
        digitalWrite(13, LOW);
        digitalWrite(11, LOW);
        digitalWrite(10, LOW);
        delay(2000);
      } else if (receivedChar == 'G') {
        digitalWrite(11, HIGH);  // Turn on Green LED
        digitalWrite(13, LOW);
        digitalWrite(12, LOW);
        digitalWrite(10, LOW);
        delay(2000);
      } else if (receivedChar == 'Y') {
        digitalWrite(10, HIGH);  // Turn on Yellow LED
        digitalWrite(13, LOW);
        digitalWrite(12, LOW);
        digitalWrite(11, LOW);
        delay(2000);
      }
    }
  }