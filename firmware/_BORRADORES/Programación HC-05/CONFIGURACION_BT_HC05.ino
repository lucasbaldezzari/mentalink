
#include<SoftwareSerial.h>
SoftwareSerial mySerial(10, 11);// son los pines de // RX, TX



void setup() {
  Serial.begin(9600);//comunicacion al pc
  mySerial.begin(38400);//comunicacion al BT
  Serial.println("configuracion BT");//MENSAJE
  
}

void loop() {
  //EN ESTA PARTE TODO LO QUE ESCRIBA SE VISUALIZA EN EL MONITOR SERIAL
  if(mySerial.available())
      Serial.write(mySerial.read());
   if(Serial.available())
    mySerial.write(Serial.read());
//EN MONITOR SERIAL SE DEBE COLOCAR CR+LF
}
