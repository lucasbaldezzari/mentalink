
#include "SoftwareSerial.h"
SoftwareSerial BT(11,10);
void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  BT.begin(9600);
  
  Serial.println("ready!");
}

void loop() {
  // put your main code here, to run repeatedly:
  if(BT.available()){
    Serial.println(BT.read());
  }
  if(Serial.available()){
    String res=(Serial.readString());
    int Mode = res.toInt();
    BT.write(Mode);
    Serial.println(Mode);
  }
}

