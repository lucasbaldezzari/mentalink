
#include "Wire.h"

uint8_t I2C_SLAVE_ADDR =0x01;

const bool ResetInError=1;

const String Arduino_Start_1 = "Canal Serial Abierto :)";
const String Arduino_Start_2 = "Canal Bluetooth Abierto :)";
const String Arduino_Start_3 = "Canal Wire Abierto :)";
const String Arduino_Start_4 = "Arduino Maesto Iniciado";

const String Arduino_OnRecive = "Recivido : ";
const String Arduino_MovimientoIniciado = "Movimiento Iniciado : ";
const String Arduino_MovimientoTerminado = "Apagando Movimiento";

const String Arduino_ERR_1 = "Err 1 : Algo salio mal cambiando el modo del arduino";
String Arduino_ERR_2 = "Err 2 : Algo salio mal añadiendo un modo al esclavo";
if(ResetInError){Arduino_ERR_2+="\nEl esclavo se Reiniciara"}


String Codigos_de_Movimiento[7][12]={
  {"0","0","0","0","0","0","0","0","1","10","","1"},
    {"1","0","1","0","0","1","0","1","4","5000","0"/Sensor usado/,"20"},
    {"0","1","0","1","1","f","1","0","6","5000","1","20"},
    {"0","1","0","1","0","1","0","1","8","5000","-1","23"},
    {"1","0","1","0","1","0","1","0","10","5000","-1","23"},
    {"1","0","1","0","0","1","0","1","12","5000","2","0"},
    {"0","1","0","1","1","0","1","0","14","5000","3","0"},
};

uint8_t Motor_Num[8]={3,4,5,6,7,8,9,10};
//uint8_t Motor_Num[8]={2,6,10,11,A0,A1,A2,A3};
uint8_t acutal_Motor_Modes[4]={0,0,0,0};
uint8_t Mode=0;



void setup(){

  long t1 = millis();
  
  
  for(int i = 0;i<8;i++){pinMode(Motor_Num[i],OUTPUT);}
  
  Wire.begin();
  
  Serial.begin(9600);
  Serial.println(Arduino_Start_1);
	
  BTSerial.begin(9600);
  BTSerial.write(Arduino_Start_2);
  
  while(true){
    if(Iniciar()){break;}
            
  }
  
  Serial.println(Arduino_Start_4);Serial.print(millis()-t1);Serial.println(" mm");
  
}
void loop(){
  if(BTSerial.available()){
  	byte res=(BTSerial.read()-'0');
    Mode = res;
    CambiarMovimiento(Mode);
    Serial.println(Arduino_OnRecive+String(Mode));
  }
}
bool Iniciar_Movimiento(uint8_t codigo){
  //Valga la redundancia Este metodo inicia un movimiento
  //Devuelve true si el movivimiento se inicio correctamente
  //Devuelve false si algo sale mal
  
  if(CambiarModo(Codigos_de_Movimiento[codigo][8].toInt()/*Cambia modo del esclavo*/)){
    //Iniciar motor 1, si algo falla devuelve falso y apaga todo
    if(!Motor(1,Codigos_de_Movimiento[codigo][0],Codigos_de_Movimiento[codigo][1])){ApagarTodo();return false;}
    //Iniciar motor 2, si algo falla devuelve falso y apaga todo
    if(!Motor(2,Codigos_de_Movimiento[codigo][2],Codigos_de_Movimiento[codigo][3])){ApagarTodo();return false;}
    //Iniciar motor 3, si algo falla devuelve falso y apaga todo
    if(!Motor(3,Codigos_de_Movimiento[codigo][4],Codigos_de_Movimiento[codigo][5])){ApagarTodo();return false;}
    //Iniciar motor 4, si algo falla devuelve falso y apaga todo
    if(!Motor(4,Codigos_de_Movimiento[codigo][6],Codigos_de_Movimiento[codigo][7])){ApagarTodo();return false;}
    
    Mode=codigo;
    return true;
    
  }else{
  	return false;
  }
}

void CambiarMovimiento(uint8_t codigo){
	//Cambiar Movimiento, Osea salir del estado de reposo a X movimiento 
  	//realizar todo su tiempo y movimiento hasta que se termine o tenga algo delante
  	//al finaliar detiene todo y vuelve a reposo
  
  if(Iniciar_Movimiento(Codigos_de_Movimiento[codigo][9].toInt())){
  	//Si logra iniciar correctamente el movimiento
    long t1 = millis();//Tiempo en el que se inicio el movimiento
    int delta=0;//No hace falta explicar lo que delta tiempo significa
    while(delta<Codigos_de_Movimiento[codigo][9].toInt()){
    	//Mientras el delta tiempo sea menor que el tiempo de ejecucion del movimeinto hace este loop
      if(Codigos_de_Movimiento[codigo][10]=="-1"){
      	//Si no usa ningun sensor
      	//No hace nada
      }else{
      	//Si usa algun sensor conectado
        uint8_t distancia= PedirDatos();
        //Pedir la distancia reportada por el sensor
        if(distancia<Codigos_de_Movimiento[codigo][11]){
          //Si la distancia es menor que el minimo permitido
          break;
        }
        long t2 = millis();
      	delta = t2-t1;
      }
    }
    ApagarTodo();
    
  }else{
  	//Si algo sale
  }
}

bool CambiarModo(uint8_t Codigo){
  Send_To_Slave(Key_Confirmar_Modo,10);
  //Pide confirmacion de en que modo esta el esclvo
  if(Obtener_Respuesta(0)!=Codigo){
    //Si no esta en el modo que se necesita
    Send_To_Slave(Key_Cambiar_Modo,Codigo);
    //Cambia el modo del esclavo
    Send_To_Slave(Key_Confirmar_Modo,10);
    //Pide confirmacion de en que modo esta el esclvo
    
    //1er intento
    if(Obtener_Respuesta(0)!=Codigo){
      //Si no esta en el modo que se necesita
      Send_To_Slave(Key_Cambiar_Modo,Codigo);
      //Cambia el modo del esclavo
      Send_To_Slave(Key_Confirmar_Modo,10);
      //Pide confirmacion de en que modo esta el esclvo
      
      //2do intento
      if(Obtener_Respuesta(0)!=Codigo){
        //Si no esta en el modo que se necesita
   	    Send_To_Slave(Key_Cambiar_Modo,Codigo);
        //Cambia el modo del esclavo
        Send_To_Slave(Key_Confirmar_Modo,10);
        if(Obtener_Respuesta(0)==Codigo){
          return true;
        }
      }else{
      	return true;
      }
    }else{
      return true;
    }
  }else{
    return true;
  }
  Serial.println(Arduino_ERR_1);
  return false;
}

void ApagarTodo(){
  CambiarModo(1);
  //Cambia el esclavo a modo reposo
  
  for(int i = 0;i<8;i++){
    digitalWrite(Motor_Num[i],false);
  	//apaga todos los motores
  }
  for(int i =0;i<4;i++){
    acutal_Motor_Modes[i]=0;
    //apaga las referencias de los motores en esta variable
  }
  Serial.println(Arduino_MovimientoTerminado);
  Mode=0;
}
bool Motor(uint8_t motor,String a,String b){
  byte _a=a.toInt();
  byte _b=b.toInt();
  byte c=a|b<<1;
  switch(c){
  	case 3:return false;
    default:acutal_Motor_Modes[motor]=c;
  }
  switch(motor){
    default:return false;
  	case 1:digitalWrite(Motor_Num[0],_a);digitalWrite(Motor_Num[1],_b);break;
    case 2:digitalWrite(Motor_Num[2],_a);digitalWrite(Motor_Num[3],_b);break;
    case 3:digitalWrite(Motor_Num[4],_a);digitalWrite(Motor_Num[5],_b);break;
    case 4:digitalWrite(Motor_Num[6],_a);digitalWrite(Motor_Num[7],_b);break;
  }
  return true;
}
bool Iniciar(){
  ApagarTodo();
  //Apaga todo para iniciar
  
  for(int i =0; i<sizeof(Codigos_de_Movimiento);i++){
    //Agrega todos los modos al esclavo
    if(!AgregarModo(i,Codigos_de_Movimiento[i][8],Codigos_de_Movimiento[i][10]),1){return false;}
  }
  
  //si todo se agredo retorna true
  return true;
}

bool AgregarModo(int pos,int a, int b, int c,bool master){
  Send_To_Slave(Key_Agregar_Modo,pos,a,b,c);
  //Manda a agregar el modo a, con el valor b y el sensor c en la posicion pos
  Send_To_Slave(Key_Confirmar_Modo,pos);
  //Pide la confirmacion del modo guardado en la posicion pos
  
  uint32 get = Obtener_Respuesta(2);
  
  byte recivedData[3];
  recivedData[0] = get;
  recivedData[1] = get >>  8;
  recivedData[2] = get >> 16;
  //Obtenemos los valores del metodo guardado en la posicion pos
  if(master){
    for(int i = 0 ; i<3;i++){
      //El metodo se llamara a si mismo 3 veces, pero las 3 repeticiones no tienen permiso
      //De llamar 3 veces a si misma porque generarian un bucle infinito y no gracias
      if(AgregarModo(pos,a,b,c,0)){return true;}
    }
    //Si sale del for y llega hasta aca significa que los 3 intentos fallaron
    //Antes de retornar falso reinicia el arduino esclavo si la variable ResetInError es true
    if(ResetInError){
    	Send_To_Slave(Key_Cambiar_Modo,Codigo);
      	EsperarQueReviva();
    }
    
    return false;
  }else{
    if(recivedData[0] =a&&recivedData[1] =b&&recivedData[2] =c){
      return true;
    }else{
      return false;
    }
  }
}
bool EsperarQueReviva(){
  while(!Wire.available()){
  	Obtener_Respuesta(0);
  }
  Wire.read();
}


uint32_t Obtener_Respuesta(int t){
  uint32_t response;
  Serial.println(sizeof(response));
  Wire.requestFrom(I2C_SLAVE_ADDR, sizeof(response));
  long t = millis();
  while(Wire.available() < 1){
    long delta = millis()-t;
    if(delta>500){
      return 0;
    }
  }
  for(int i = 0;i<((8*t)+1);i+=8){
  	response =response |( Wire.read()<<i);
  }
  return response;
}
void Send_To_Slave(uint8_t data){
  Wire.beginTransmission(I2C_SLAVE_ADDR);
  Wire.write((byte*)&data, sizeof(data));
  Wire.endTransmission();
}
void Send_To_Slave(uint8_t data,uint8_t data1){
  Wire.beginTransmission(I2C_SLAVE_ADDR);
  Wire.write((byte*)&data, sizeof(data));
  Wire.write((byte*)&data1, sizeof(data1));
  Wire.endTransmission();
}
void Send_To_Slave(uint8_t data,uint8_t data1,uint8_t data2){
  Wire.beginTransmission(I2C_SLAVE_ADDR);
  Wire.write((byte*)&data, sizeof(data));
  Wire.write((byte*)&data1, sizeof(data1));
  Wire.write((byte*)&data2, sizeof(data2));
  Wire.endTransmission();
}
void Send_To_Slave(uint8_t data,uint8_t data1,uint8_t data2,uint8_t data3){
  Wire.beginTransmission(I2C_SLAVE_ADDR);
  Wire.write((byte*)&data, sizeof(data));
  Wire.write((byte*)&data1, sizeof(data1));
  Wire.write((byte*)&data2, sizeof(data2));
  Wire.write((byte*)&data3, sizeof(data3));
  Wire.endTransmission();
}
void Send_To_Slave(uint8_t data,uint8_t data1,uint8_t data2,uint8_t data3,uint8_t data4){
  Wire.beginTransmission(I2C_SLAVE_ADDR);
  Wire.write((byte*)&data, sizeof(data));
  Wire.write((byte*)&data1, sizeof(data1));
  Wire.write((byte*)&data2, sizeof(data2));
  Wire.write((byte*)&data3, sizeof(data3));
  Wire.write((byte*)&data4, sizeof(data4));
  Wire.endTransmission();
}