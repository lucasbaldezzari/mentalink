#include<SoftwareSerial.h>
SoftwareSerial BTEsclavo(10,11);

int LEDAd= 2;
int LEDDe= 3;
int LEDRD= 4;
int LEDAt= 5;
int LEDRI= 6;         
int LEDIz= 7;

int DATO=0;
byte Ad = 0b00000001;
byte De = 0b00000010;
byte RD = 0b00000100;
byte At = 0b00001000;
byte RI = 0b00010000;
byte Iz = 0b00100000;


void setup(){
  pinMode(LEDAd,OUTPUT);
  pinMode(LEDDe,OUTPUT);
  pinMode(LEDRD,OUTPUT);
  pinMode(LEDAt,OUTPUT);
  pinMode(LEDRI,OUTPUT);
  pinMode(LEDIz,OUTPUT);
  
  Serial.begin(9600);
  BTEsclavo.begin(9600);
  delay(100);
  BTEsclavo.listen();
  delay(100);
  }

void loop() 
  {
    //if(BTEsclavo.isListening()) digitalWrite(led,1);
  if(BTEsclavo.available())
  {
    DATO=BTEsclavo.read();
    if (DATO == Ad) digitalWrite(LEDAd,1);
    if (DATO == De) digitalWrite(LEDDe,1);
    if (DATO == RD) digitalWrite(LEDRD,1);
    if (DATO == At) digitalWrite(LEDAt,1);
    if (DATO == RI) digitalWrite(LEDRI,1);
    if (DATO == Iz) digitalWrite(LEDIz,1);
    //Serial.println(DATO);
    }
    
//   if(DATO=='0'){
//    digitalWrite(led,LOW);
//    Serial.println("OF");
//   }
//   if(DATO=='1'){
//    digitalWrite(led,HIGH);
//    Serial.println("OF");
//   }
  }
