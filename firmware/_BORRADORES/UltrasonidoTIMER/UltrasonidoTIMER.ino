#include "inicializaciones.h"

const int Trigger = 12;   //Pin digital 2 para el Trigger del sensor
const int Echo = 13;   //Pin digital 3 para el Echo del sensor
long t; //timepo que demora en llegar el eco
long d; //distancia en centimetros
char restTime; //variable para controlar los eco
int contador = 100; //contador para que no se sature el ultrasonido
int LED = 3; //led para prueba

void setup() {
  Serial.begin(9600);//iniciailzamos la comunicación
  pinMode(Trigger, OUTPUT); //pin como salida
  pinMode(Echo, INPUT);  //pin como entrada
  digitalWrite(Trigger, LOW);//Inicializamos el pin con 0
  pinMode(LED, OUTPUT);
  iniTimer2(); //inicio timer 2
}

void loop()
{
}

ISR(TIMER2_COMPA_vect)//Rutina interrupción Timer2, configurado a 10us
{
  if (contador == 100){ //para que no se sature agrego 100ms entre ciclos
    if (restTime = 'OFF'){ //si se debe enviar el eco
      digitalWrite(Trigger, HIGH);
      restTime = 'ON'; //para que de un descanso de 10us antes de leer el eco
    }
    if (restTime = 'ON'){ //si se esta esperando el eco
      digitalWrite(Trigger, LOW);
      t = pulseIn(Echo, HIGH); //obtenemos el ancho del pulso
      d = t/59;             //escalamos el tiempo a una distancia en cm
      Serial.print("Distancia: ");
      Serial.print(d);      //Enviamos serialmente el valor de la distancia
      Serial.print("cm");
      Serial.println();
      contador = 0; //reinicio el contador para que al finalizar este if comience a contar 100ms
      restTime = 'OFF'; //si se debe enviar el eco
      if(d<15) digitalWrite(LED, 1); //si la distancia es menor a 15 cm encender el LED
      else digitalWrite(LED, 0);
    } 
    }  
  contador++; //suma al contador cada vez que se genera la interrupcion
};
