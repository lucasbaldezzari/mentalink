#include "inicializaciones.h"
#include<SoftwareSerial.h>

const int Trigger1 = 12;   //Pin digital
const int Echo1 = 13;   //Pin digital
const int Trigger2 = 6;   //Pin digital
const int Echo2 = 7;   //Pin digital
long t1; //timepo que demora en llegar el eco
long d1; //distancia en centimetros
long t2; //timepo que demora en llegar el eco
long d2; //distancia en centimetros
char restTime; //variable para controlar los eco
int contador1 = 50000; //contador para que no se sature el ultrasonido
int contador2 = 100000;
int contador = 0;
int LED1 = 3; //led para prueba
int LED2 = 4; //led para prueba
int LEDtest = 2;
byte obstaculos = 0b00000000;
int estado = 0;
byte recepcion = 0b00000000;

SoftwareSerial BTEsclavo(10,9); //RX, TX (1-10) (0-9) en Serial M3.1 

void setup() {
  Serial.begin(9600);//iniciailzamos la comunicación
  BTEsclavo.begin(9600);
  pinMode(Trigger1, OUTPUT); //pin como salida
  pinMode(Echo1, INPUT);  //pin como entrada
  pinMode(Trigger2, OUTPUT); //pin como salida
  pinMode(Echo2, INPUT);  //pin como entrada
  digitalWrite(Trigger1, 0);//Inicializamos el pin con 0
  digitalWrite(Trigger2, 0);//Inicializamos el pin con 0
  pinMode(LED1, OUTPUT);
  pinMode(LED2, OUTPUT);
  pinMode(LEDtest, OUTPUT);
  iniTimer2(); //inicio timer 2
  iniTimer1();
  interrupts();//Habilito las interrupciones
}

void loop()
{
}

ISR(TIMER1_COMPA_vect)//Rutina interrupción Timer1, configurado a 10us
{
  if (BTEsclavo.available()){
    BTEsclavo.write(obstaculos);
    obstaculos = 0b00000000;
    estado = !estado;
    digitalWrite(LEDtest, estado);
    recepcion = BTEsclavo.read();
  }
}

ISR(TIMER2_COMPA_vect)//Rutina interrupción Timer2, configurado a 10us
{

  if (contador == 5000){ //para que no se sature agrego 50ms entre ciclos
    if (restTime = 'OFF'){ //si se debe enviar el eco
      digitalWrite(Trigger1, HIGH);
      restTime = 'ON'; //para que de un descanso de 10us antes de leer el eco
    }
    if (restTime = 'ON'){ //si se esta esperando el eco
      digitalWrite(Trigger1, LOW);
      t1 = pulseIn(Echo1, HIGH); //obtenemos el ancho del pulso
      d1 = t1/59;             //escalamos el tiempo a una distancia en cm
//      Serial.print("Distancia: ");
//      Serial.print(d1);      //Enviamos serialmente el valor de la distancia
//      Serial.print("cm");
//      Serial.println();
      restTime = 'OFF'; //si se debe enviar el eco
      if(d1<15){ 
        digitalWrite(LED1, 1);
        obstaculos = obstaculos | 0b00000001;
      }//si la distancia es menor a 15 cm encender el LED
     if(obstaculos == 0b00000000) digitalWrite(LED1, 0);
    } 
    }  
//  if (contador == 10000){ //para que no se sature agrego 50ms entre ciclos
//    if (restTime = 'OFF'){ //si se debe enviar el eco
//      digitalWrite(Trigger2, HIGH);
//      restTime = 'ON'; //para que de un descanso de 10us antes de leer el eco
//    }
//    if (restTime = 'ON'){ //si se esta esperando el eco
//      digitalWrite(Trigger2, LOW);
//      t2 = pulseIn(Echo2, HIGH); //obtenemos el ancho del pulso
//      d2 = t2/59;             //escalamos el tiempo a una distancia en cm
////      Serial.print("Distancia: ");
////      Serial.print(d2);      //Enviamos serialmente el valor de la distancia
////      Serial.print("cm");
////      Serial.println();
//      contador = 0; //reinicio el contador para que al finalizar este if comience a contar 50ms
//      restTime = 'OFF'; //si se debe enviar el eco
//      if(d2<15){ 
//        digitalWrite(LED2, 1);
//        obstaculos = obstaculos | 0b00000010;
//        }
//      if(obstaculos == 0b00000000) digitalWrite(LED2, 0); 
//    }
//  } 
  contador++; //suma al contador cada vez que se genera la interrupcion
}
//
//void SendMenssage(byte mensaje){
//  BTEsclavo.write(mensaje);
//}
//}
