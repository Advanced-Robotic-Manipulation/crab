/*
 * Pedal Controller for Crab Robot
 * D2 = turn_left, D3 = turn_right, D4 = go_forward, D5 = go_back
 */

#define PIN_TURN_LEFT   2
#define PIN_TURN_RIGHT  3
#define PIN_GO_FORWARD  4
#define PIN_GO_BACK     5

#define NUM_PEDALS 4
const int PEDAL_PINS[NUM_PEDALS] = {PIN_TURN_LEFT, PIN_TURN_RIGHT, PIN_GO_BACK, PIN_GO_FORWARD};
const char* PEDAL_NAMES[NUM_PEDALS] = {"turn_left", "turn_right", "go_back", "go_forward"};

#define DEBOUNCE_MS 10
unsigned long lastDebounce[NUM_PEDALS] = {0, 0, 0, 0};
bool pedalState[NUM_PEDALS] = {false, false, false, false};
bool lastReading[NUM_PEDALS] = {false, false, false, false};

void setup() {
    Serial.begin(115200);
    
    for (int i = 0; i < NUM_PEDALS; i++) {
        pinMode(PEDAL_PINS[i], INPUT_PULLUP);
    }
    
    pinMode(LED_BUILTIN, OUTPUT);
    Serial.println("PEDALS_READY");
}

void loop() {
    bool anyPressed = false;
    unsigned long now = millis();
    
    for (int i = 0; i < NUM_PEDALS; i++) {
        bool reading = (digitalRead(PEDAL_PINS[i]) == LOW);
        
        if (reading != lastReading[i]) {
            lastDebounce[i] = now;
        }
        
        if ((now - lastDebounce[i]) > DEBOUNCE_MS) {
            pedalState[i] = reading;
        }
        
        lastReading[i] = reading;
        if (pedalState[i]) anyPressed = true;
    }
    
    digitalWrite(LED_BUILTIN, anyPressed ? HIGH : LOW);
    
    if (Serial.available() > 0) {
        String cmd = Serial.readStringUntil('\n');
        cmd.trim();
        
        if (cmd == "read" || cmd == "r") {
            sendState();
        } else if (cmd == "ping") {
            Serial.println("pong");
        }
    }
}

void sendState() {
    // Output order: turn_left, turn_right, go_back, go_forward
    // Pin order:    D2,        D3,         D5,      D4
    Serial.print("turn_left:");
    Serial.print(pedalState[0] ? "1" : "0");
    Serial.print(",turn_right:");
    Serial.print(pedalState[1] ? "1" : "0");
    Serial.print(",go_back:");
    Serial.print((digitalRead(PIN_GO_BACK) == LOW) ? "1" : "0");
    Serial.print(",go_forward:");
    Serial.println((digitalRead(PIN_GO_FORWARD) == LOW) ? "1" : "0");
}
