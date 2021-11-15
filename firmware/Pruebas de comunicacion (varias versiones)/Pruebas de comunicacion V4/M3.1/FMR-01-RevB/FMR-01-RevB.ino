#include <SoftwareSerial.h>
#include "inicializaciones.h"

#define   FRENADO    1
#define   MOVIENDO   0

SoftwareSerial BTone(3, 4);  // TX, RX

char Dt = 0;

char ledRojo = 11;
char ledVerde = 12;
char ledAzul = 13;

int temporizador = 500; //para 100ms cFRENADOsiderando la frecuencia de interrupción del timer2
int acum = 0;
bool estado = 0;
byte obstaculos = 0b00000000;

char flagMoviendo = FRENADO;

byte mascaraComando = 0b00000111;

unsigned int acumulador = 0;

// Motor A
int ENA = 5;
int IN1 = 6;
int IN2 = 7;

// Motor B
int ENB = 10;
int IN3 = 8;
int IN4 = 9;


void setup() {
  noInterrupts();//Deshabilito todas las interrupciFRENADOes
  //Motores
  pinMode (ENA, OUTPUT);
  pinMode (ENB, OUTPUT);
  pinMode (IN1, OUTPUT);
  pinMode (IN2, OUTPUT);
  pinMode (IN3, OUTPUT);
  pinMode (IN4, OUTPUT);
  pinMode (ledRojo, OUTPUT);
  pinMode (ledVerde, OUTPUT);
  pinMode (ledAzul, OUTPUT);
  pinMode (2, OUTPUT);
  digitalWrite(2, 0);
  analogWrite (ENA, 0); //motores parados
  analogWrite (ENB, 0); //motores parados
  BTone.begin(9600);
  Serial.begin(9600);

  iniTimer2();
  delay(1000);
  digitalWrite(ledRojo, 0);
  interrupts();//Habilito las interrupciFRENADOes
}

void loop() {
  //if (BTone.available()) digitalWrite(2,1);
  }

/*Rutina interrupciòn Timer0*/
ISR(TIMER2_COMPA_vect)//Rutina interrupción Timer2
{
  if (BTone.available())
  {
    estado = !estado;
    digitalWrite(ledVerde,estado);
    Dt = BTone.read();    
    giveAnOrder();//damos una orden al vehículo
    //sendBTMessage();
    Serial.write(BTone.read());   
  }

  if (Serial.available())
  {
    obstaculos = Serial.read();
    if (obstaculos != 0b00000000){
      Stop();
      digitalWrite(2,1);
    }
    else digitalWrite(2,0);
    BTone.write(obstaculos);
  }

  switch (flagMoviendo)
  {
    case MOVIENDO:
      if (++acum > temporizador)
      {
        estado = !estado;
        digitalWrite(ledRojo, estado);
        acum = 0;
      }
      break;

    case FRENADO:
      if (++acum > temporizador)
      {
        estado = !estado;
        digitalWrite(ledAzul, estado);
        acum = 0;
      }
      break;
  }

};

void Adelante()
{
  analogWrite (ENA, 100);
  analogWrite (ENB, 100);
  digitalWrite (IN1, LOW);
  digitalWrite (IN2, HIGH);
  digitalWrite (IN3, HIGH);
  digitalWrite (IN4, LOW);
}

void Retroceso()
{
  analogWrite (ENA, 100);
  analogWrite (ENB, 100);
  digitalWrite (IN1, HIGH );
  digitalWrite (IN2, LOW);
  digitalWrite (IN3, LOW );
  digitalWrite (IN4, HIGH);
}

void Derecha()
{
  analogWrite (ENA, 0);
  analogWrite (ENB, 100);
  digitalWrite (IN1, LOW );
  digitalWrite (IN2, LOW );
  digitalWrite (IN3, HIGH);
  digitalWrite (IN4, LOW);
}

void Izquierda()
{
  analogWrite (ENA, 100);
  analogWrite (ENB, 0);
  digitalWrite (IN1, LOW);
  digitalWrite (IN2, HIGH);
  digitalWrite (IN3, LOW );
  digitalWrite (IN4, LOW );
}

void Stop()
{
  digitalWrite (IN1, LOW);
  digitalWrite (IN2, LOW);
  digitalWrite (IN3, LOW);
  digitalWrite (IN4, LOW);
  analogWrite (ENA, 0); //motores parados
  analogWrite (ENB, 0); //motores parados
}

void giveAnOrder()
{
  if ( ( (Dt >> 1) & 0b00000001) == 0 ) //Si los estímulos se apagarFRENADO, podemos mover.
  {
    flagMoviendo = MOVIENDO;
    digitalWrite(ledAzul, 0);
    acum = 0;
    if ( ((Dt >> 2) & mascaraComando) == 1) Adelante();
    else if ( ((Dt >> 2) & mascaraComando) == 2) Izquierda();
    else if ( ((Dt >> 2) & mascaraComando) == 3) Retroceso();
    else if ( ((Dt >> 2) & mascaraComando) == 4 ) Derecha();
    else    {
      Stop();
    }
  }

  else
  {
    Stop();
    flagMoviendo = FRENADO;
    digitalWrite(ledRojo, 0);
    acum = 0;
  }

  if ( ( (Dt >> 0) & 0b00000001) == 0 ) //Sesión frenada
  {
    Stop();
    flagMoviendo = FRENADO;
    digitalWrite(ledRojo, 0);
    acum = 0;
  }

}

//void sendBTMessage()
//{
//  BTone.write(0b00000010);
//}
