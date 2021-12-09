#include <SoftwareSerial.h>
#include "inicializaciones.h"

#define   FRENADO    1
#define   MOVIENDO   0

SoftwareSerial BTone(A3, A4);  // TX, RX

char Dt = 0;

char ledRojo = A0;
char ledVerde = A5;
char ledAzul = A1;

int temporizador = 500; //para 100ms cFRENADOsiderando la frecuencia de interrupción del timer2
int acum = 0;
bool estado = 0;
byte obstaculos = 0b00000000;

char flagMoviendo = FRENADO;

byte mascaraComando = 0b00000111;

unsigned int acumulador = 0;

// Motor A
int ENA = 10;
int IN1 = 6;
int IN2 = 7;

// Motor B
int ENB = 10;
int IN3 = 8;
int IN4 = 9;

// Motor C
int ENC = 10;
int IN5 = 4;
int IN6 = 2;

// Motor D
int END = 10;
int IN7 = 12;
int IN8 = 13;

void setup() {
  noInterrupts();//Deshabilito todas las interrupciFRENADOes
  //Motores
  pinMode (ENA, OUTPUT);
  pinMode (ENB, OUTPUT);
  pinMode (ENC, OUTPUT);
  pinMode (END, OUTPUT);
  pinMode (IN1, OUTPUT);
  pinMode (IN2, OUTPUT);
  pinMode (IN3, OUTPUT);
  pinMode (IN4, OUTPUT);
  pinMode (IN5, OUTPUT);
  pinMode (IN6, OUTPUT);
  pinMode (IN7, OUTPUT);
  pinMode (IN8, OUTPUT);
  pinMode (ledRojo, OUTPUT);
  pinMode (ledVerde, OUTPUT);
  pinMode (ledAzul, OUTPUT);
  analogWrite (ENA, 0); //motores parados
  analogWrite (ENB, 0); //motores parados
  analogWrite (ENC, 0); //motores parados
  analogWrite (END, 0); //motores parados
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
    }
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
  analogWrite (ENA, 200);
  analogWrite (ENB, 200);
  digitalWrite (IN1, LOW);
  digitalWrite (IN2, HIGH);
  digitalWrite (IN3, HIGH);
  digitalWrite (IN4, LOW);
  analogWrite (ENC, 200);
  analogWrite (END, 200);
  digitalWrite (IN5, LOW);
  digitalWrite (IN6, HIGH);
  digitalWrite (IN7, HIGH);
  digitalWrite (IN8, LOW);
}

void Retroceso()
{
  analogWrite (ENA, 200);
  analogWrite (ENB, 200);
  digitalWrite (IN1, HIGH );
  digitalWrite (IN2, LOW);
  digitalWrite (IN3, LOW );
  digitalWrite (IN4, HIGH);
  analogWrite (ENC, 200);
  analogWrite (END, 200);
  digitalWrite (IN5, HIGH);
  digitalWrite (IN6, LOW);
  digitalWrite (IN7, LOW);
  digitalWrite (IN8, HIGH);
}

void Derecha()
{
  analogWrite (ENA, 255);
  analogWrite (ENB, 255);
  digitalWrite (IN1, LOW); //atras izquierda
  digitalWrite (IN2, HIGH);
  digitalWrite (IN3, LOW); //atras derecha
  digitalWrite (IN4, HIGH);
  analogWrite (ENC, 255);
  analogWrite (END, 255);
  digitalWrite (IN5, HIGH); //adelante derecha
  digitalWrite (IN6, LOW);
  digitalWrite (IN7, HIGH); //adelante izquierda
  digitalWrite (IN8, LOW);
}

void Izquierda()
{
  analogWrite (ENA, 255);
  analogWrite (ENB, 255);
  digitalWrite (IN1, HIGH);
  digitalWrite (IN2, LOW);
  digitalWrite (IN3, HIGH );
  digitalWrite (IN4, LOW);
  analogWrite (ENC, 255);
  analogWrite (END, 255);
  digitalWrite (IN5, LOW);
  digitalWrite (IN6, HIGH);
  digitalWrite (IN7, LOW);
  digitalWrite (IN8, HIGH);
}

void Stop()
{
  digitalWrite (IN1, LOW);
  digitalWrite (IN2, LOW);
  digitalWrite (IN3, LOW);
  digitalWrite (IN4, LOW);
  analogWrite (ENA, 0); //motores parados
  analogWrite (ENB, 0); //motores parados
  analogWrite (ENC, 0);
  analogWrite (END, 0);
  digitalWrite (IN5, LOW);
  digitalWrite (IN6, LOW);
  digitalWrite (IN7, LOW);
  digitalWrite (IN8, LOW);
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
    else if ( ((Dt >> 2) & mascaraComando) == 4) Retroceso();
    else if ( ((Dt >> 2) & mascaraComando) == 3 ) Derecha();
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
