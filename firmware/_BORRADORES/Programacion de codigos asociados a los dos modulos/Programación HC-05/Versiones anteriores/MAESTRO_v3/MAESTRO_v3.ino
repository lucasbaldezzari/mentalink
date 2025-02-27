/******************************************************************
            VERSIÓN FMR-001 Rev A
******************************************************************/

#include "definiciones.h"
#include "inicializaciones.h"
#include "funciones.h"
#include<SoftwareSerial.h>

/******************************************************************
  Declaración de variables para comunicación
/******************************************************************/

char inBuffDataFromPC = 3;
unsigned char incDataFromPC[3]; //variable para almacenar datos provenientes de la PC
char bufferIndex = 0;
bool sendDataFlag = 0;
bool newMessage = false;

unsigned char internalStatus[4]; //variable para enviar datos a la PC
char internalStatusBuff = 4;

unsigned char outputDataToRobot[4]; //variable para enviar datos al robot
char buffOutDataRobotSize = 4;
char buffOutDataRobotIndex = 0;

unsigned char incDataFromRobot[4]; //variable para recibir datos del robot
char incDataFromRobotSize = 4;
char incDataFromRobotIndex = 0;

/******************************************************************
  Variables para el control de flujo de programa
******************************************************************/
char sessionState = 0; //Sesión sin iniciar
char LEDVerde = 13; 
char LEDTesteo = 5; //led de testeo

/******************************************************************
  Declaración de variables para control de estímulos
******************************************************************/

int frecTimer = 5000; //en Hz. Frecuencia de interrupción del timer.

//estímulo izquierdo
char estimIzq = 4;
bool estimIzqON = 0;//Esado que define si el LED se apgará o prenderá.
int frecEstimIzq = 11;
int acumEstimIzq = 0;
const int estimIzqMaxValue = (1/float(frecEstimIzq))*frecTimer;

//estímulo derecho
char estimDer = 12;
bool estimDerON = 0;//Esado que define si el LED se apgará o prenderá.
int frecEstimDer = 3;
int acumEstimDer = 0;
const int estimDerMaxValue = (1/float(frecEstimDer))*frecTimer;

/*Implementar lo siguiente
//estímulo adelante
//estímulo atrás
*/

char stimuli = ON; //variable golbal para control de estímulos
int acumuladorStimuliON = 0;
int trialNumber = 1;

/******************************************************************
  Variables control de movimiento
******************************************************************/

char movimiento = 0; //Robot en STOP

SoftwareSerial BTMaestro(10,9); //RX, TX

//FUNCION SETUP
void setup()
{
  noInterrupts();//Deshabilito todas las interrupciones
  pinMode(estimIzq,OUTPUT);
  pinMode(estimDer,OUTPUT);
  pinMode(LEDVerde,OUTPUT);
  pinMode(LEDTesteo,OUTPUT);
  pinMode(3,OUTPUT);
  iniTimer0(frecTimer); //inicio timer 0
  Serial.begin(19200); //iniciamos comunicación serie
  BTMaestro.begin(9600);//comunicacion al BT
  interrupts();//Habilito las interrupciones
}

void loop(){}

void serialEvent()
{
if (Serial.available() > 0) 
  {
    char val = char(Serial.read()) - '0';
    checkMessage(val); //chequeamos mensaje entrante        
    for(int index = 0; index < internalStatusBuff; index++) //enviamos estado 
      {Serial.write(internalStatus[index]);}
      Serial.write("\n");
  }
};

ISR(TIMER0_COMPA_vect)//Rutina interrupción Timer0.
{
  if(sessionState == SESSION_RUNNING) stimuliControl(); //Si la sesión comenzó, empezamos a generar los estímulos  
  else
  {
    //apago estímulos
    digitalWrite(estimIzq,0);
    digitalWrite(estimDer,0);
    digitalWrite(LEDTesteo,1);
  }
};

void stimuliControl()
{
  acumuladorStimuliON++;
  switch(stimuli)
  {
    case ON:
    //control estímulo izquierdo
      if (++acumEstimIzq >= estimIzqMaxValue)
      {
        estimIzqON = !estimIzqON;
        digitalWrite(estimIzq,estimIzqON);
        acumEstimIzq = 0; 
      } 

    //control estímulo derecho
      if (++acumEstimDer >= estimDerMaxValue)
      {
        estimDerON = !estimDerON;
        digitalWrite(estimDer,estimDerON);
        acumEstimDer = 0; 
      } 
      break;

    case OFF:
      {
        //Apagamos estímulos y reiniciamos contadores
        digitalWrite(estimIzq,0);
        acumEstimIzq = 0;
        digitalWrite(estimDer,0);
        acumEstimDer = 0;
        trialNumber++; //Sumamos un nuevo trial
        acumuladorStimuliON = 0; //reiniciamos acumulador para temporizar cada trial
        acumEstimIzq = 0;
        acumEstimDer = 0;
      } 
      break;
  }
}

void checkMessage(byte val)
{
  incDataFromPC[bufferIndex] = val;
  switch(bufferIndex)
  {
    case 0:
      sendCommand(val);
      if (incDataFromPC[bufferIndex] == 1) sessionState = SESSION_RUNNING;
      else sessionState = SESSION_STOP;
      digitalWrite(LEDTesteo,0);
      
      //else sessionState = STOP;
      break;
    case 1:
      sendCommand(val);
      if (incDataFromPC[bufferIndex] == 1) stimuli = ON;
      else stimuli = OFF;
      
      break;

    case 2: //indica hacia donde se debe mover el vehículo
      sendCommand(val);
      break;          
  }
  bufferIndex++;
  if (bufferIndex >= inBuffDataFromPC) bufferIndex = 0;
};

/*
Función: sendCommand()
- Se usa para enviar un comando al vehículo robótico y recibir el estado del mismo.
*/
void sendCommand(byte comando)
{
  BTMaestro.write(comando);
}
