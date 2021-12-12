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

int Detenido = 2;
int Adelante = 3;
int Atras = 4;
int Derecha = 5;
int Izquierda = 6;
int RotD = 7;
int RotI = 8;
int SESSION = 13;


byte DATO = 0;

SoftwareSerial BTEsclavo(10, 11);// son los pines de // RX, TX conectados a arduino

void setup(){
  pinMode(Detenido,OUTPUT);
  pinMode(Adelante,OUTPUT);
  pinMode(Atras,OUTPUT);
  pinMode(Derecha,OUTPUT);
  pinMode(Izquierda,OUTPUT);
  pinMode(RotD,OUTPUT);
  pinMode(RotI,OUTPUT);
  pinMode(SESSION,OUTPUT);
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
    Control(DATO);
    }
  }  

void Movimiento(byte val)
{
  switch(val)
  {
//    case 0b00000000: 
//      digitalWrite(Detenido, 1);
//      break;
//    case 0b00000001: 
//      digitalWrite(Detenido, 1);
//      break;
    case 0: 
      digitalWrite(Detenido, 1);
      break;
    case 1: 
    digitalWrite(Adelante, 1);
      break;
    case 2: 
      digitalWrite(Atras, 1);
      break;
    case 3: 
      digitalWrite(Derecha, 1);
      break;
    case 4: 
      digitalWrite(Izquierda, 1);
      break;
    case 5: 
      digitalWrite(RotD, 1);
      break;
    case 6: 
      digitalWrite(RotI, 1);
      break;
  }
};

void Control(byte DATO)
{
  byte Session = ((DATO)&0b00000001);
  byte Stimulo = ((DATO>>1)&0b00000001);
  Serial.println(Stimulo);
  byte Orden = ((DATO>>2)&0b00000111);
//    Movimiento(Orden);
    if ((Stimulo == 0)) 
    {Movimiento(Orden);
    
    }
    if (Stimulo == 1){
      digitalWrite(Detenido, 1);
      digitalWrite(Adelante, 0);
      digitalWrite(Atras, 0);
      digitalWrite(Derecha, 0);
      digitalWrite(Izquierda, 0);
      digitalWrite(RotD, 0);
      digitalWrite(RotI, 0);
    }
    }



//void Movimiento(byte orden)
//{
//  switch(orden)
//  {
//    case 0: 
//      if (digitalRead(Detenido)) digitalWrite(Detenido, 0);
//      else digitalWrite(Detenido, 1);
//      break;
//    case 1: 
//      digitalWrite(Adelante, 1);
//      break;
//    case 2: 
//      digitalWrite(Atras, 1);
//      break;
//    case 3: 
//      digitalWrite(Derecha, 1);
//      break;
//    case 4: 
//      digitalWrite(Izquierda, 1);
//      break;
//    case 5: 
//      digitalWrite(RotD, 1);
//      break;
//    case 6: 
//      digitalWrite(RotI, 1);
//      break;
//  }
//  
//}
