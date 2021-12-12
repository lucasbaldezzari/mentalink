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

volatile bool Cambiando_Modo=false;
volatile bool Agregando_Modo=false;
volatile bool Target_Modo_seleccionado=false;
volatile int  Target=0;
volatile bool Modo_keycode_annadida=false;
volatile int  keycode=0;
volatile bool Modo_variable_val_annadido=false;
volatile int  variable=0;
volatile byte active=0;
volatile bool Pidiendo_Modo=false;
volatile int  Modo_Solicitado=0;
volatile bool Pidiendo_Modo_Val=false;
volatile byte ModoOff_V=0;
volatile byte Modo0ff[2]={0,0};
volatile int ModoDebug_V=0;
volatile bool Enviado_1=false;
volatile bool Enviado_2=false;
volatile bool Enviado_3=false;


volatile byte ModoDebug[2]={1,1};
//*******************************/


//*******************************/
volatile int ModoAdelante_V=0;
volatile byte ModoAdelante[2]={2,0};

volatile int ModoAtras_V=0;
volatile byte ModoAtras[2]={3,0};

volatile int ModoGirando_V=0;
volatile byte ModoGirando[2]={4,0};
//********************************/

int Sensor_Pines[2][2]={{2,3},{5,6}};


volatile bool Error_Ocurred=false;

volatile byte Modo = 0;

byte entra=0; //Valor de entrada
byte CODE; //distancia del objeto y envio
int suma =0;

bool Haciendo_Algo = 0;
bool Active = 1;


void setup() {
 	//CODIGO esclavo
  	Serial.begin(9600);
    if(TestearSensores()){

      Wire.begin(0x01); //identifico como esclavo 1
  	  Wire.onReceive( receiveEvent); //Declaro Evento
 	    Wire.onRequest(Peticion); //Declaro Evento
      pinMode(1, OUTPUT);
    }else{
      Serial.println("Los Sensores estan desconectados y/o Cargaste el codigo del esclavo en el maestro");
      Active =0; 
    }



}

void loop() {
  if (Modo== 0){Serial.println("I am here");Reset();}
  if(Active){

      
      if (Modo== 1){digitalWrite(1, 0);}
      if (Modo== ModoAdelante[0]||ModoAdelante_V==0){digitalWrite(1, 1);Analizar(2);}
      if (Modo== ModoAtras[0]||ModoAtras_V==0){digitalWrite(1, 1);Analizar(3);}
      if (Modo== ModoGirando[0]){digitalWrite(1, 1);Analizar(4);}
  }

}
bool TestearSensores(){
  long has=Analizar(Sensor_Pines[0][0],Sensor_Pines[0][1]);
  if(has=0){
    has=Analizar(Sensor_Pines[0][0],Sensor_Pines[0][1]);
    if(has=0){
      return false;
    }
  }
   has=Analizar(Sensor_Pines[1][0],Sensor_Pines[1][1]);
  if(has=0){
    has=Analizar(Sensor_Pines[1][0],Sensor_Pines[1][1]);
    if(has=0){
      return false;
    }
  }
  return true;
}

void Reset(){
    Wire.endTransmission();

	Serial.println("Reset");
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
    ModoGirando[1]=0;
    Modo =1;Wire.begin(0x01); //identifico como esclavo 1
 	Wire.onReceive( receiveEvent); //Declaro Evento
 	Wire.onRequest(Peticion); //Declaro Evento
}

void receiveEvent(int bytes){
    byte data = 0;
    uint8_t index = 0;
    while (Wire.available()){
        byte* pointer = (byte*)&data;
        *(pointer + index) = (byte)Wire.read();
        index++;
    }
   
    if(Haciendo_Algo){
      
      if(Cambiando_Modo){
        if(
        data ==ModoGirando[0]||
        data ==ModoAtras[0]||
        data ==ModoAdelante[0]||
        data ==1 ||data==0
        ){
          Modo=data;
          Cambiando_Modo=false;
          Haciendo_Algo=0;
          Error_Ocurred=false;
        }else{
          Error_Ocurred=true;
          Cambiando_Modo=false;
        }
      }else if(Agregando_Modo){
        if(Error_Ocurred){
          Target=0;
          Target_Modo_seleccionado=false;
          keycode=0;
          Modo_keycode_annadida=false;
          variable=0;
          Modo_variable_val_annadido=false;
          active=0;
          Haciendo_Algo=0;
          Agregando_Modo=false;
        }else if(Target_Modo_seleccionado){
          if(Modo_keycode_annadida){
            if(Modo_variable_val_annadido){
              if(data==1||data==0){
                active=data;
                  switch(Target){
                    case 2:Change_Adelante(keycode,variable,active);break;
                    case 3:Change_Atras(keycode,variable,active);break;
                    case 4:Change_Girando(keycode,variable,active);break;
                  }
                Agregando_Modo=false;
                Target=0;
                Haciendo_Algo=0;
                Target_Modo_seleccionado=false;
                keycode=0;
                Modo_keycode_annadida=false;
                variable=0;
                Modo_variable_val_annadido=false;
                active=0;
                Error_Ocurred=false;
              }else{
                Agregando_Modo=false;
                Error_Ocurred=true;Haciendo_Algo=0;
              }        
            }else{
              variable=data;
              Modo_variable_val_annadido=true;
            }
          }else{
            if(
              data !=ModoGirando[0] and
              data !=ModoAtras[0] and
              data !=ModoAdelante[0] and
              data !=1 and data!=0
            ){
              keycode=data;
              Modo_keycode_annadida=true;
              Error_Ocurred=false;
            }else{
              if(data==get_Keycode(Target)){
             	  keycode=data;
                Modo_keycode_annadida=true;
                Error_Ocurred=false;
              }else{
             	  Error_Ocurred=true;Haciendo_Algo=0;
              }
            }
          }
        }else{
          if(data >1 and data<5){
            Target=data;
            Target_Modo_seleccionado=true;
            Error_Ocurred=false;
          }else{
            Error_Ocurred=true;Haciendo_Algo=0;
            Target_Modo_seleccionado=false;
          }     
        }
      }
      else if(Pidiendo_Modo ){
        if(
        data ==ModoGirando[0]||
        data ==ModoAtras[0]||
        data ==ModoAdelante[0]||
        data ==1 ||data==0||data==5
        ){
          Modo_Solicitado=data;
          Serial.println(data);
          Haciendo_Algo=0;
        }else{Pidiendo_Modo=false;Haciendo_Algo=0;}
      }
    }else{
      switch(data){
            case Key_Cambiar_Modo:Cambiando_Modo=true;Haciendo_Algo=1;break;
            case Key_Confirmar_Modo:Pidiendo_Modo=true;Haciendo_Algo=1;break;
            case Key_Agregar_Modo:Agregando_Modo=true;Haciendo_Algo=1;break;
            case Key_Confirmar_Variable:Pidiendo_Modo_Val=true;break;
    }
  }
}
void Peticion(){
    if(Error_Ocurred){
        Wire.write(0); 
        Error_Ocurred=0;
      	
    }else{
      
        if(Pidiendo_Modo){
          if(Modo_Solicitado==ModoAdelante[0]){
          	if(Enviado_1){
              if(Enviado_2){
                Serial.println("Enviando : "+ String(ModoAdelante[1]));
                Wire.write(ModoAdelante[1]);
                Enviado_1=false;Enviado_2=false;
              }else{
                Serial.println("Enviando : "+ String(ModoAdelante_V));
                Wire.write(ModoAdelante_V);
                Enviado_2=true;
              }
            }else{
            Serial.println("Enviando : "+ String(ModoAdelante[0]));
              Wire.write(ModoAdelante[0]);
              Enviado_1=true;
            }
          }else if(Modo_Solicitado==ModoAtras[0]){
          	if(Enviado_1){
              if(Enviado_2){
                Serial.println("Enviando : "+ String(ModoAtras[1]));
                Wire.write(ModoAtras[1]);
                Enviado_1=false;
                Enviado_2=false;
                }else{
                Wire.write(ModoAtras_V);
                Serial.println("Enviando : "+ String(ModoAtras_V));
                Enviado_2=true;
                }
              }else{
                Wire.write(ModoAtras[0]);
                Serial.println("Enviando : "+ String(ModoAtras[0]));
                Enviado_1=true;
              }
          }else if(Modo_Solicitado==ModoGirando[0]){
            if(Enviado_1){
              if(Enviado_2){
              Wire.write(ModoGirando[1]); 
              Enviado_1=false;
              Enviado_2=false;
            }else{
              Wire.write(ModoGirando_V);
             Enviado_2=true;
              }
            }else{
              Wire.write(ModoGirando[0]);
              Enviado_1=true;
            }
          }else switch(Modo_Solicitado){
                case 0:Wire.write(0);break;
                case 1:Wire.write(1);break;
              	case 5:Wire.write(Modo);break;
            Serial.println("Enviando modo : "+String(Modo));break;
                default:Error_Ocurred=!true;
            }
          	if(!Enviado_1 and !Enviado_2 and !Enviado_3){
            	Modo_Solicitado=0;
            	Pidiendo_Modo=false;
          	}
          
        }else if(Pidiendo_Modo_Val){
          	if(Modo==0){Wire.write(0);}
            if(Modo==1){Wire.write(0);}
            if(Modo==ModoAdelante[0]){Wire.write(ModoAdelante_V);Serial.println("Dis : "+String(ModoAdelante_V));}
          	if(Modo==ModoAtras[0]){Wire.write(ModoAtras_V);Serial.println("Dis : "+String(ModoAtras_V));}
          	if(Modo==ModoGirando[0]){Wire.write(ModoGirando_V);}
          	Pidiendo_Modo_Val=false;
        }

    }
    
}
void Change_Adelante(int key, int val, byte active){
    ModoAdelante_V=val;
	ModoAdelante[0]=key;
  	ModoAdelante[1]=active;
}
void Change_Atras(int key, int val, byte active){
    ModoAtras_V=val;
	ModoAtras[0]=key;
  	ModoAtras[1]=active;
}
void Change_Girando(int key, int val, byte active){
    ModoGirando_V=val;
    ModoGirando[0]=key;
  	ModoGirando[1]=active;
}
byte get_Keycode(int target){
  	switch(target){
  		case 0:return 0;
      	case 1:return 1;
      	case 2:return ModoAdelante[0];
      	case 3:return ModoAtras[0];
      	case 4:return ModoGirando[0];
      	default:Error_Ocurred=true;return 0;
    }

}
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
long Analizar(int triggerPin, int echoPin)
{
  pinMode(triggerPin, OUTPUT);  // Clear the trigger
  digitalWrite(triggerPin, LOW);
  delayMicroseconds(2);
  // Sets the trigger pin to HIGH state for 10 microseconds
  digitalWrite(triggerPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(triggerPin, LOW);
  pinMode(echoPin, INPUT);
  // Reads the echo pin, and returns the sound wave travel time in microseconds
  return pulseIn(echoPin, HIGH);
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
}