#include<SoftwareSerial.h>

SoftwareSerial mySerial(10, 11);// son los pines de // RX, TX

byte bit1Snact = 0b00000000;
byte bit1Sact = 0b00000001;
byte bit2OFF = 0b00000000;
byte bit2ON = 0b0000010;
byte bitM1 = 0b00000100; //Adelante
byte bitM2 = 0b00001000; //Atrás
byte bitM3 = 0b00001100; //Derecha
byte bitM4 = 0b00010000; //Izquierda

void setup() {
  Serial.begin(9600);//comunicacion al pc
  mySerial.begin(9600);//comunicacion al BT
}

void loop() {
 mySerial.write((bit1Sact)|(bit2OFF)|(bitM1));
  Serial.println((bit1Sact)|(bit2OFF)|(bitM1));
  delay(2000);
  mySerial.write((bit1Sact)|(bit2OFF)|(bitM2));
  Serial.println((bit1Sact)|(bit2OFF)|(bitM2));
  delay(2000);
  mySerial.write((bit1Sact)|(bit2OFF)|(bitM3));
  Serial.println((bit1Sact)|(bit2OFF)|(bitM3));
  delay(2000);
  mySerial.write((bit1Sact)|(bit2OFF)|(bitM4));
  Serial.println((bit1Sact)|(bit2OFF)|(bitM4));
  delay(2000);
  mySerial.write((bit1Sact)|(bit2ON));
  Serial.println((bit1Sact)|(bit2ON));
  delay(2000);
}
