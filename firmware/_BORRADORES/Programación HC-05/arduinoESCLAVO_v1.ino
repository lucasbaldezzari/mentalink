#include<SoftwareSerial.h>
SoftwareSerial BTEsclavo(10,9); //TX, RX

int LEDAd= 12;
int LEDAt= 13;
int LEDDe= 4;
int LEDIz = 5;

int DATO=0;

byte bit1Snact = 0b00000000;
byte bit1Sact = 0b00000001;
byte bit2OFF = 0b00000000;
byte bit2ON = 0b0000010;
byte bitM1 = 0b00000100; //Adelante
byte bitM2 = 0b00001000; //Atrás
byte bitM3 = 0b00001100; //Derecha
byte bitM4 = 0b00010000; //Izquierda

void setup(){
  pinMode(LEDAd,OUTPUT);
  Serial.begin(9600);
  BTEsclavo.begin(9600);
  delay(100);
  BTEsclavo.listen();
  delay(100);
  }

void loop() 
  {
  //if(BTEsclavo.isListening()) digitalWrite(LEDAd,1);
  if(BTEsclavo.available());
  {
    DATO=BTEsclavo.read();
    Serial.println(DATO);
    if (DATO == ((bit1Sact)|(bit2OFF)|(bitM1))) digitalWrite(LEDAd,1);
    if (DATO == ((bit1Sact)|(bit2OFF)|(bitM2))) digitalWrite(LEDAt,1);
    if (DATO == ((bit1Sact)|(bit2OFF)|(bitM3))) digitalWrite(LEDDe,1);
    if (DATO == ((bit1Sact)|(bit2OFF)|(bitM4))) digitalWrite(LEDIz,1);
    if (DATO == ((bit1Sact)|(bit2ON))){
      digitalWrite(LEDAd,0);
      digitalWrite(LEDAt,0);
      digitalWrite(LEDDe,0);
      digitalWrite(LEDIz,0);
    }
    }
   }
 
