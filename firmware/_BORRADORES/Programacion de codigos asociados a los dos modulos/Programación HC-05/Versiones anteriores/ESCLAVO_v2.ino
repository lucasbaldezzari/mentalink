#include<SoftwareSerial.h>
SoftwareSerial BTEsclavo(11, 10);// son los pines de // RX, TX

int ledStop = 13;

int DATO = 0;

byte bitSTOP = 0b00000000;

void setup(){
  pinMode(ledStop,OUTPUT);
  Serial.begin(9600);
  BTEsclavo.begin(9600);
  delay(100);
  BTEsclavo.listen();
  delay(100);
  }

void loop() 
  {
//  if(BTEsclavo.isListening()) digitalWrite(ledStop,1);
  if(BTEsclavo.available());
  {
    DATO=BTEsclavo.read();
    Serial.println(DATO);
    if (DATO == 0) digitalWrite(ledStop,1);
    }
    
   }
 
