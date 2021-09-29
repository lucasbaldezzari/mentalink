/******************************************************************
            VERSIÓN FMR-001 Rev A
******************************************************************/
#include "SoftwareSerial.h"
#include "definiciones.h"
#include "inicializaciones.h"
#include "funciones.h"

/******************************************************************
  Declaración de variables para comunicación
/******************************************************************/

char inBuffDataFromPC = 3;
byte incDataFromPC[3]; //variable para almacenar datos provenientes de la PC
/*
- incDataFromPC[0]: Estado de sesión (RUNNING = 1, STOP = 0)
- incDataFromPC[1]: Estado trial (ON = 1, OFF = 0)
- incDataFromPC[2]: Comando (0=adelante, 1=atras...)
*/
char bufferIndex = 0;
bool sendDataFlag = 0;
bool newMessage = false;

byte robotStatus[3]; //variable para enviar datos a la PC
char robotStatusBuff = 3;

byte outputDataToRobot[4]; //variable para enviar datos al robot
char buffOutDataRobotSize = 4;
char buffOutDataRobotIndex = 0;

byte incDataFromRobot[4]; //variable para recibir datos del robot
char incDataFromRobotSize = 4;
char incDataFromRobotIndex = 0;

/******************************************************************
  Variables para el control de flujo de programa
******************************************************************/
char sessionState = 0; //Sesión sin iniciar
char LEDVerde =13; 
char LEDTesteo = 12; //led de testeo

/******************************************************************
  Declaración de variables para control de estímulos
******************************************************************/

int frecTimer = 5000; //en Hz. Frecuencia de interrupción del timer.

//estímulo izquierdo
char estimIzq = 5;
bool estimIzqON = 0;//Esado que define si el LED se apgará o prenderá.
int frecEstimIzq = 11;
int acumEstimIzq = 0;
const int estimIzqMaxValue = (1/float(frecEstimIzq))*frecTimer;

//estímulo derecho
char estimDer = 4;
bool estimDerON = 0;//Esado que define si el LED se apgará o prenderá.
int frecEstimDer = 11;
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
char backMensaje = 0;
/******************************************************************
  Bluetooth
******************************************************************/
SoftwareSerial BT(11,10); //(RX||TX)

//FUNCION SETUP
void setup()
{
  noInterrupts();//Deshabilito todas las interrupciones
  pinMode(estimIzq,OUTPUT);
  pinMode(estimDer,OUTPUT);
  pinMode(LEDVerde,OUTPUT);
  pinMode(LEDTesteo,OUTPUT);
  iniTimer0(frecTimer); //inicio timer 0
  Serial.begin(19200); //iniciamos comunicación serie
  BT.begin(9600);//iniciamos comunicación Bluetooth
  delay(1000);
  interrupts();//Habilito las interrupciones
}

void loop(){}

void serialEvent()
{
if (Serial.available() > 0) 
  {
    char val = (Serial.read()) - '0';
    checkSerialMessage(val); //chequeamos mensaje entrante        
    for(int index = 0; index < robotStatusBuff; index++) //enviamos estado 
      {
        Serial.write(robotStatus[index]);
      }
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
//
//  if(BT.available()) //Si tenemos un mensaje por bluetooth lo leemos
//  {
//    backMensaje = BT.read();
//    checkBTMessage(backMensaje);
//  }
};

void stimuliControl()
{
  acumuladorStimuliON++;
  switch(stimuli)
  {
    case ON:
    //control estímulo izquierda
      if (++acumEstimIzq >= estimIzqMaxValue)
      {
        estimIzqON = !estimIzqON;
        if ((backMensaje&0b00100000) == 0b00100000) digitalWrite(estimIzq,0);
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

void checkSerialMessage(char val)
{
  incDataFromPC[bufferIndex] = val;
  switch(bufferIndex)
  {
    case 0:
      if (incDataFromPC[bufferIndex] == 1) sessionState = SESSION_RUNNING;
      else sessionState = SESSION_STOP;
      digitalWrite(LEDTesteo,0);
      //else sessionState = STOP;
      break;
    case 1:
      if (incDataFromPC[bufferIndex] == 1) stimuli = ON;
      else stimuli = OFF;
      //else sessionState = STOP;
      break;

    case 2: //indica hacia donde se debe mover el vehículo
      byte comando = incDataFromPC[bufferIndex];
      if(comando == 3) digitalWrite(LEDVerde,1);
       //Es mejor hacer sendCommand(incDataFromPC[bufferIndex])
      break;          
  }
  bufferIndex++;
  if (bufferIndex >= inBuffDataFromPC) //hemos recibido todos los bytes desde la PC
  {
    sendMensajeBT();
    bufferIndex = 0;
  }
};

//void checkBTMessage(char val)
//{
//}


/*
Función: sendMensajeBT()
- Se usa para enviar un comando al vehículo robótico a través de Bluetooth
*/
void sendMensajeBT()
{
    /*
    i) mensaje = (0b00000001)|(0b00000000<<1)|(0b00000100<<2)
    ii) mensaje = (0b00000001)|(0b00000000)|(0b00100000)
    iii) mensaje = 0b00100001
    */
    
    byte mensaje = (incDataFromPC[0])|(incDataFromPC[1]<<1)|(incDataFromPC[2]<<2);//Armamos el byte
    BT.write(mensaje); //enviamos byte por bluetooth
    
}
