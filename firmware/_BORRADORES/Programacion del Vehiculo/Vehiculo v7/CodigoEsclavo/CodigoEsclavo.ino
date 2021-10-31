//Version 1.1

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
#include "Wire.h" 


const int Key_Cambiar_Modo = 200;
const int Key_Confirmar_Modo = 211;
const int Key_Agregar_Modo = 222;
const int Key_Confirmar_Variable=233;

const int Error_code =0;

const String Advertencia_Sensor_Desconectado="Advertencia: Es posible que halla un sensor desconectado";
const String Alerta_Sensor_Desconectado="Alerta: Es posible que halla un sensor desconectado o los codigos esten cambiados\n  Conecte los sensores y reinicie el arduino.";

volatile bool Cambiando_Modo=false;
volatile bool Agregando_Modo=false;
//volatile bool Target_Modo_seleccionado=false;
volatile int  Target=0;
//volatile bool Modo_keycode_annadida=false;
volatile int  keycode=0;
//volatile bool Modo_variable_val_annadido=false;
volatile int  variable=0;
//volatile byte active=0;
volatile bool Pidiendo_Modo=false;
volatile int  Modo_Solicitado=0;
volatile bool Pidiendo_Modo_Val=false;
//volatile byte ModoOff_V=0;
//volatile byte Modo0ff[2]={0,0};
//volatile int ModoDebug_V=0;
//volatile bool Enviado_1=false;
//volatile bool Enviado_2=false;
//volatile bool Enviado_3=false;

//volatile byte ModoDebug[2]={1,1};
//*******************************/


//*******************************/
//key sensor value
volatile byte Modos[5][3]={
  {2,0,40},
  {3,1,40},
  {4,2,40},
  {5,3,40},
  {6,3,40}
};
//********************************/

byte AccionSolicitada = 0;

bool Sensores_Conectados[4]={1,1,0,0};

int Sensor_Pines[4][2]={{2,3},{5,6},{2,3},{5,6}};

volatile bool Error_Ocurred=false;

volatile byte Modo = 0;

bool Haciendo_Algo = 0;
bool Active = 1;


void setup() {
 	//CODIGO esclavo
  	Serial.begin(9600);
    if(TestearSensores()){
      Wire.begin(0x01); //identifico como esclavo 1
  	  //Wire.onReceive( receiveEvent); //Declaro Evento
 	    Wire.onRequest(Peticion); //Declaro Evento
      pinMode(1, OUTPUT);
      Active=1;
    }else{
      Serial.println(Alerta_Sensor_Desconectado);
      Active =0; 
    }
}

void loop() {
  if(Wire.available()){receiveEvent();}
  if (Modo== 0){Serial.println("I am here");Reset();}
  else if(Modo==1){TestearSensores();}
  else if(Active){
    if(Sensores_Conectados[Modo-2]){
      Modos[Modo-2][2]=0.01723 *Analizar(Sensor_Pines[Modo-2][0],Sensor_Pines[Modo-2][1]);
    }else{
      Modos[Modo-2][2]=40;
    }
      
  }else{
    Error_Ocurred=1;
  }
}
bool TestearSensores(){
  for(int i = 0; i<sizeof(Sensores_Conectados);i++){
    if(Sensores_Conectados[i]){
      if(Analizar(Sensor_Pines[i][0],Sensor_Pines[i][1])<=0){
        Serial.println("Intento 1 del senosor "+String(i+1)+" Fallido");
        if(Analizar(Sensor_Pines[i][0],Sensor_Pines[i][1])<=0){
          Serial.println("Intento 2 del senosor "+String(i+1)+" Fallido");
          if(Analizar(Sensor_Pines[i][0],Sensor_Pines[i][1])<=0){
            Serial.println("Intento 3 del senosor "+String(i+1)+" Fallido");
            if(!Active){
              Serial.println(Advertencia_Sensor_Desconectado);
            }
            return false;

          }
        }
      }
    }
  }
  return true;
}

void Reset(){
  Wire.endTransmission();
	Serial.println("Reset");/*
  Cambiando_Modo=false;
  Agregando_Modo=false;
  Target_Modo_seleccionado=false;
  Target=0;
  Modo_keycode_annadida=false;
  keycode=0;
  Modo_variable_val_annadido=false;
  variable=0;
  active=0;
  Pidiendo_Modo=false;
  Modo_Solicitado=0;
  Pidiendo_Modo_Val=false;   
  ModoAdelante_V=0;
  ModoAdelante[0]=0;
	ModoAdelante[1]=0;
  
  ModoAtras_V=0;
  ModoAtras[0]=0;
	ModoAtras[1]=0;

  ModoGirando_V=0;
  ModoGirando[0]=0;
  ModoGirando[1]=0;*/
  Modo =1;Wire.begin(0x01); //identifico como esclavo 1
 	//Wire.onReceive( receiveEvent); //Declaro Evento
 	Wire.onRequest(Peticion); //Declaro Evento
}

void receiveEvent(){
  byte data = 0;
  while(Wire.available() < 1){}
  uint8_t in = Wire.read();
  byte AccionEnviada = in & 0b00000011;
  switch(AccionEnviada){
    case 0:{//Pedir modo val
      AccionSolicitada=(AccionEnviada+1);
    }break;
    case 1:{//Cambiar Modo
      byte ValorSecundario = (in>>2)&0b00001111;//Valor real
      int i = 0;
      for(;i<sizeof(Modos);i++){
        if(ValorSecundario==Modos[i][0]){
          Modo = i+2;
          break; 
        }
        i++;
      }
      Error_Ocurred=1;
    }break;
    case 2:{//confirmacion de modo
      byte ValorSecundario = (in>>2)&0b00001111;//Valor real
      int i = 0;
      for(;i<sizeof(Modos);i++){
        if(ValorSecundario==Modos[i][0]){
          AccionSolicitada=(AccionEnviada);
          Modo_Solicitado=i;
          break; 
        }
        if(ValorSecundario==10){
          AccionSolicitada=(AccionEnviada+1);
          Modo_Solicitado=5;
      }
        i++;
      }
      Error_Ocurred=1;
    }break;
    case 3:{
      byte ValorSecundario = (in>>2)&0b00000111;//Valor real
      byte ValorTerciario  = (in>>5);//Key

      while(Wire.available() < 1){}
      in = Wire.read();
      byte ValorCuaternario = in & 0b00000111;
      byte ValorUltimo =  (in>>3);
      Modos[ValorSecundario][0]=ValorTerciario;
      Modos[ValorSecundario][1]=ValorCuaternario;
      Modos[ValorSecundario][2]=ValorUltimo;
    }break;

  }


  
}
void Peticion(){
  if(Error_Ocurred){
    Wire.write(0); 
    Error_Ocurred=0;
  }else{
    switch(AccionSolicitada){
      case 1:{
        if(Modo_Solicitado>10){Wire.write(0);break;}
        if(Modo_Solicitado==10){Wire.write(Modos[Modo-2][2]);break;}
        Wire.write(Modos[Modo_Solicitado][2]);
      }break;
      case 2:{
        byte msj = Modos[Modo_Solicitado][0] | Modos[Modo_Solicitado][1]<<4 ;
        Wire.write(msj|Modos[Modo_Solicitado][2]<<8);
      }break;
    }
  }
  AccionSolicitada=0;
  Modo_Solicitado=0;    
}
long Analizar(int triggerPin, int echoPin){
  pinMode(triggerPin, OUTPUT);  // Clear the trigger
  digitalWrite(triggerPin, LOW);
  delayMicroseconds(2);
  //Sets the trigger pin to HIGH state for 10 microseconds
  digitalWrite(triggerPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(triggerPin, LOW);
  pinMode(echoPin, INPUT);
  // Reads the echo pin, and returns the sound wave travel time in microseconds
  return pulseIn(echoPin, HIGH);
}
/*
void Analizar(int _target){
 	switch(_target){
    case 0:break ;
    case 1:break ;
    case 2:set_ModoVal(0.01723 * Analizar(Sensor_Pines[0][0],Sensor_Pines[0][1]),2);break;
    case 3: set_ModoVal(0.01723 * Analizar(Sensor_Pines[1][0],Sensor_Pines[1][1]),3);break;
    case 4:set_ModoVal(ModoGirando_V,4);
    default:Error_Ocurred=true;
   }
}

void set_ModoVal(long val,int pos){
	switch(pos){
		case 0:break ;
   	case 1:break ;
  	case 2:ModoAdelante_V=val;break ;
   	case 3:ModoAtras_V=val;break ;
  	case 4:ModoGirando_V=val;break ;
   	default:Error_Ocurred=true;break ;
  }
}*/