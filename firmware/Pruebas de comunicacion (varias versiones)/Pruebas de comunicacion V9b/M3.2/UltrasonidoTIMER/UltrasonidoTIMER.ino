#include "inicializaciones.h"
#include<SoftwareSerial.h>

const int TriggerAd = 5;   //Pin digital
const int EchoAd = 4;   //Pin digital
const int TriggerAt = 12;   //Pin digital
const int EchoAt = 11;   //Pin digital
const int TriggerDer = A0;   //Pin digital
const int EchoDer = A1;   //Pin digital
const int TriggerIz = A4;   //Pin digital
const int EchoIz = A5;   //Pin digital
long t1; //timepo que demora en llegar el eco
long d1; //distancia en centimetros
long t2; //timepo que demora en llegar el eco
long d2; //distancia en centimetros
long t3; //timepo que demora en llegar el eco
long d3; //distancia en centimetros
long t4; //timepo que demora en llegar el eco
long d4; //distancia en centimetros
char restTime; //variable para controlar los eco
int contador1 = 50000; //contador para que no se sature el ultrasonido
int contador2 = 100000;
int contador = 0;
int LEDAd = 6; //led para prueba
int LEDAt = 13; //led para prueba
int LEDDer = A2; //led para prueba
int LEDIz = A3; //led para prueba
//int LEDtest = A2;
byte obstaculos = 0b00000000;
int estado = 0;
byte recepcion = 0b00000000;
int flagDetenido = 0;
int flagVel1 = 0;
int flagVel2 = 0;

//SoftwareSerial BTEsclavo(10,9); //RX, TX (1-10) (0-9) en Serial M3.1

void setup() {
  Serial.begin(9600);//iniciailzamos la comunicación
  //BTEsclavo.begin(9600);
  pinMode(TriggerAd, OUTPUT); //pin como salida
  pinMode(EchoAd, INPUT);  //pin como entrada
  pinMode(TriggerAt, OUTPUT); //pin como salida
  pinMode(EchoAt, INPUT);  //pin como entrada
  pinMode(TriggerDer, OUTPUT); //pin como salida
  pinMode(EchoDer, INPUT);  //pin como entrada
  pinMode(TriggerIz, OUTPUT); //pin como salida
  pinMode(EchoIz, INPUT);  //pin como entrada
  digitalWrite(TriggerAd, 0);//Inicializamos el pin con 0
  digitalWrite(TriggerAt, 0);//Inicializamos el pin con 0
  digitalWrite(TriggerDer, 0);//Inicializamos el pin con 0
  digitalWrite(TriggerIz, 0);//Inicializamos el pin con 0
  pinMode(LEDAd, OUTPUT);
  pinMode(LEDAt, OUTPUT);
  pinMode(LEDDer, OUTPUT);
  pinMode(LEDIz, OUTPUT);
  //pinMode(LEDtest, OUTPUT);
  iniTimer2(); //inicio timer 2
  iniTimer1();
  interrupts();//Habilito las interrupciones
}

void loop()
{
}

//ISR(TIMER1_COMPA_vect)//Rutina interrupción Timer1, configurado a 10us
//{
//  if (Serial.available()) {
//    Serial.write(obstaculos);
//    obstaculos = 0b00000000;
//    estado = !estado;
//    //digitalWrite(LEDtest, estado);
//    recepcion = Serial.read();
//  }
//}

//void serialEvent(){
//  if (Serial.available()) {
//    Serial.write(obstaculos);
//    obstaculos = 0b00000000;
//    estado = !estado;
//    //digitalWrite(LEDtest, estado);
//    recepcion = Serial.read();
//  }
//}

ISR(TIMER2_COMPA_vect)//Rutina interrupción Timer2, configurado a 10us
{  
  
if (Serial.available()) {
  Serial.write(obstaculos);
  obstaculos = 0b00000000;
  estado = !estado;
  //digitalWrite(LEDtest, estado);
  recepcion = Serial.read();
}

//if ((recepcion & 0b00000010) == 0b00000000){    
  if (contador == 2000) { //para que no se sature agrego 20ms entre ciclos
    if (restTime = 'OFF') { //si se debe enviar el eco
      digitalWrite(TriggerAd, HIGH);
      restTime = 'ON'; //para que de un descanso de 10us antes de leer el eco
    }
    if (restTime = 'ON') { //si se esta esperando el eco
      digitalWrite(TriggerAd, LOW);
      t1 = pulseIn(EchoAd, HIGH); //obtenemos el ancho del pulso
      d1 = t1 / 59;           //escalamos el tiempo a una distancia en cm
      //      Serial.print("Distancia: ");
      //      Serial.print(d1);      //Enviamos serialmente el valor de la distancia
      //      Serial.print("cm");
      //      Serial.println();
      //contador = 0;
      restTime = 'OFF'; //si se debe enviar el eco
      if ((d1 < 15) && (flagDetenido == 0)) {
        digitalWrite(LEDAd, 1);
        obstaculos = obstaculos | 0b00000001;
        Serial.write(obstaculos);
        flagDetenido = 1;
        flagVel1 = 0;
        flagVel2 = 0;
      }
      else if ((d1 < 30) && (flagVel1 == 0) && (flagDetenido == 0)){
        digitalWrite(LEDAd, 1);
        obstaculos = obstaculos | 0b00000010;
        Serial.write(obstaculos);
        flagDetenido = 0;
        flagVel1 = 1;
        flagVel2 = 0;
      }
      else if ((d1 < 50) && (flagVel1 == 0) && (flagVel2 == 0) && (flagDetenido == 0)){
        digitalWrite(LEDAd, 1);
        obstaculos = obstaculos | 0b00000011;
        Serial.write(obstaculos);
        flagDetenido = 0;
        flagVel1 = 0;
        flagVel2 = 1;
      }
      else {
        digitalWrite(LEDAd, 0);
        obstaculos = 0b00000000;
        Serial.write(obstaculos);
        flagDetenido = 0;
        flagVel1 = 0;
        flagVel2 = 0;
      }
      
//      if (obstaculos == 0b00000000) digitalWrite(LEDAd, 0);
    }
  }
  if (contador == 4000) { //para que no se sature agrego 20ms entre ciclos
    if (restTime = 'OFF') { //si se debe enviar el eco
      digitalWrite(TriggerAt, HIGH);
      restTime = 'ON'; //para que de un descanso de 10us antes de leer el eco
    }
    if (restTime = 'ON') { //si se esta esperando el eco
      digitalWrite(TriggerAt, LOW);
      t2 = pulseIn(EchoAt, HIGH); //obtenemos el ancho del pulso
      d2 = t2 / 59;           //escalamos el tiempo a una distancia en cm
      //      Serial.print("Distancia: ");
      //      Serial.print(d2);      //Enviamos serialmente el valor de la distancia
      //      Serial.print("cm");
      //      Serial.println();
      //contador = 0; //reinicio el contador para que al finalizar este if comience a contar 50ms
      restTime = 'OFF'; //si se debe enviar el eco
       if ((d2 < 15) && (flagDetenido == 0)) {
        digitalWrite(LEDAt, 1);
        obstaculos = obstaculos | 0b00000100;
        Serial.write(obstaculos);
        flagDetenido = 1;
        flagVel1 = 0;
        flagVel2 = 0;
      }
      else if ((d2 < 30) && (flagVel1 == 0) && (flagDetenido == 0)){
        digitalWrite(LEDAt, 1);
        obstaculos = obstaculos | 0b00001000;
        Serial.write(obstaculos);
        flagDetenido = 0;
        flagVel1 = 1;
        flagVel2 = 0;
      }
      else if ((d2 < 50) && (flagVel1 == 0) && (flagVel2 == 0) && (flagDetenido == 0)){
        digitalWrite(LEDAt, 1);
        obstaculos = obstaculos | 0b00001100;
        Serial.write(obstaculos);
        flagDetenido = 0;
        flagVel1 = 0;
        flagVel2 = 1;
      }
      else {
        digitalWrite(LEDAt, 0);
        obstaculos = 0b00000000;
        Serial.write(obstaculos);
        flagDetenido = 0;
        flagVel1 = 0;
        flagVel2 = 0;
      }
//      if (obstaculos == 0b00000000) digitalWrite(LEDAt, 0);
    }
  }
   if (contador == 6000) { //para que no se sature agrego 50ms entre ciclos
    if (restTime = 'OFF') { //si se debe enviar el eco
      digitalWrite(TriggerDer, HIGH);
      restTime = 'ON'; //para que de un descanso de 10us antes de leer el eco
    }
    if (restTime = 'ON') { //si se esta esperando el eco
      digitalWrite(TriggerDer, LOW);
      t3 = pulseIn(EchoDer, HIGH); //obtenemos el ancho del pulso
      d3 = t3 / 59;           //escalamos el tiempo a una distancia en cm
      //      Serial.print("Distancia: ");
      //      Serial.print(d1);      //Enviamos serialmente el valor de la distancia
      //      Serial.print("cm");
      //      Serial.println();
      //contador = 0;
      restTime = 'OFF'; //si se debe enviar el eco
      if ((d3 < 15) && (flagDetenido == 0)) {
        digitalWrite(LEDDer, 1);
        obstaculos = obstaculos | 0b00010000;
        Serial.write(obstaculos);
        flagDetenido = 1;
        flagVel1 = 0;
        flagVel2 = 0;
      }
      if ((d3 < 30) && (flagVel1 == 0) && (flagDetenido == 0)){
        digitalWrite(LEDDer, 1);
        obstaculos = obstaculos | 0b00100000;
        Serial.write(obstaculos);
        flagDetenido = 0;
        flagVel1 = 1;
        flagVel2 = 0;
      }
      if ((d3 < 50) && (flagVel1 == 0) && (flagVel2 == 0) && (flagDetenido == 0)){
        digitalWrite(LEDDer, 1);
        obstaculos = obstaculos | 0b00110000;
        Serial.write(obstaculos);
        flagDetenido = 0;
        flagVel1 = 0;
        flagVel2 = 1;
      }
      if (d3 > 50) {
        digitalWrite(LEDDer, 0);
        obstaculos = 0b00000000;
        Serial.write(obstaculos);
        flagDetenido = 0;
        flagVel1 = 0;
        flagVel2 = 0;
      }
      }//si la distancia es menor a 15 cm encender el LED
//      if (obstaculos == 0b00000000) digitalWrite(LEDDer, 0);
    }
   if (contador == 8000) { //para que no se sature agrego 50ms entre ciclos
    if (restTime = 'OFF') { //si se debe enviar el eco
      digitalWrite(TriggerIz, HIGH);
      restTime = 'ON'; //para que de un descanso de 10us antes de leer el eco
    }
    if (restTime = 'ON') { //si se esta esperando el eco
      digitalWrite(TriggerIz, LOW);
      t4 = pulseIn(EchoIz, HIGH); //obtenemos el ancho del pulso
      d4 = t4 / 59;           //escalamos el tiempo a una distancia en cm
      //      Serial.print("Distancia: ");
      //      Serial.print(d1);      //Enviamos serialmente el valor de la distancia
      //      Serial.print("cm");
      //      Serial.println();
      contador = 0;
      restTime = 'OFF'; //si se debe enviar el eco
    if ((d4 < 15) && (flagDetenido == 0)) {
        digitalWrite(LEDIz, 1);
        obstaculos = obstaculos | 0b01000000;
        Serial.write(obstaculos);
        flagDetenido = 1;
        flagVel1 = 0;
        flagVel2 = 0;
      }
      else if ((d4 < 30) && (flagVel1 == 0) && (flagDetenido == 0)){
        digitalWrite(LEDIz, 1);
        obstaculos = obstaculos | 0b10000000;
        Serial.write(obstaculos);
        flagDetenido = 0;
        flagVel1 = 1;
        flagVel2 = 0;
      }
      else if ((d4 < 50) && (flagVel1 == 0) && (flagVel2 == 0) && (flagDetenido == 0)){
        digitalWrite(LEDIz, 1);
        obstaculos = obstaculos | 0b11000000;
        Serial.write(obstaculos);
        flagDetenido = 0;
        flagVel1 = 0;
        flagVel2 = 1;
      }
      else {
        digitalWrite(LEDIz, 0);
        obstaculos = 0b00000000;
        Serial.write(obstaculos);
        flagDetenido = 0;
        flagVel1 = 0;
        flagVel2 = 0;
      }
  }
  contador++; //suma al contador cada vez que se genera la interrupcion
}
}
//}
//
//void SendMenssage(byte mensaje){
//  BTEsclavo.write(mensaje);
//}
