#include "Wire.h"
#include <SoftwareSerial.h>       
SoftwareSerial BTSerial(12,13);

uint8_t I2C_SLAVE_ADDR =0x01;



int Codigos_de_Movimiento[7][12]={
  {0,0,0,0,0,0,0,0,500000,7,1,1},
      //[0-7]pos motores [8]tiempo [9]sensor [10]distancia minima [11]algo adelante
      {1,0,1,0,0,1,0,1,5000,0,31,1},
      {0,1,0,1,1,0,1,0,5000,1,31,1},
      {0,1,0,1,0,1,0,1,5000,7,23,1},
      {1,0,1,0,1,0,1,0,5000,7,23,1},
      {1,0,1,0,0,1,0,1,5000,2,20,1},
      {0,1,0,1,1,0,1,0,5000,3,20,1}
};


uint8_t Motor_Num[8]={2,6,10,11,A0,A1,A2,A3};

byte MovimientoActual=0;

void setup(){
  for(int i = 0;i<8;i++){pinMode(Motor_Num[i],OUTPUT);}
  Wire.begin();
  Serial.begin(9600);
  BTSerial.begin(9600);
}

void loop(){
  if(Serial.available()){
  //if(BTSerial.available()){
    
    MovimientoActual=Serial.read()-'0';
    //MovimientoActual=BTSerial.read();
    for(int i=1;i<7;i++){
      Codigos_de_Movimiento[i][11]=1;
      //Luego de enviarlo marca como que se puede mover
          //ya luego si mas adelante no se puede mover
          //volvera a setearla como 0
    }
    long TiempoInicioMovimiento = millis();
    int TiempoTranscurrido=0;
    //Serial.println("Iniciando : "+String(MovimientoActual)+"por "+String(Codigos_de_Movimiento[MovimientoActual][8])+"segs");
    while(TiempoTranscurrido<Codigos_de_Movimiento[MovimientoActual][8]){
      if(Serial.available()){
    //if(BTSerial.available()){
        
        MovimientoActual=Serial.read();
        //MovimientoActual=BTSerial.read();
        
        TiempoInicioMovimiento = millis();
        TiempoTranscurrido=0;
        
        Serial.println(BTOut());
        //BTSerial.write(BTOut());
        for(int i=1;i<7;i++){
          Codigos_de_Movimiento[i][11]=1;
          //Luego de enviarlo marca como que se puede mover
          //ya luego si mas adelante no se puede mover
          //volvera a setearla como 0
        }
      }
      byte msj = Codigos_de_Movimiento[MovimientoActual][9]|Codigos_de_Movimiento[MovimientoActual][10]<<3;

    Send_To_Slave(msj);
      Codigos_de_Movimiento[MovimientoActual][11]= Obtener_Respuesta();
      if(Codigos_de_Movimiento[MovimientoActual][11]!=1){break;}
      int t =0;
      for(int _Motor=1;_Motor<5;_Motor++){
        Motor(_Motor,Codigos_de_Movimiento[MovimientoActual][t],Codigos_de_Movimiento[MovimientoActual][t+1]);
        t+=2;
      }
      TiempoTranscurrido = millis()-TiempoInicioMovimiento;
    }
    MovimientoActual=0;
    int t =0;
    for(int _Motor=1;_Motor<5;_Motor++){
      Motor(_Motor,Codigos_de_Movimiento[MovimientoActual][t],Codigos_de_Movimiento[MovimientoActual][t+1]);
      t+=2;
    }
    Serial.println(BTOut());
    //BTSerial.write(BTOut());
  }
}



byte BTOut(){
  byte res=0;
  for(int i = 1;i<7;i++){
  res = res | Codigos_de_Movimiento[i][11]<<i;
  }
  return res;
}

bool Motor(uint8_t motor,uint8_t a,uint8_t b){
  byte c=a|b<<1;
  switch(c){
    case 3:return false;
  }
  switch(motor){
    default:return false;
    case 1:digitalWrite(Motor_Num[0],a);digitalWrite(Motor_Num[1],b);break;
    case 2:digitalWrite(Motor_Num[2],a);digitalWrite(Motor_Num[3],b);break;
    case 3:digitalWrite(Motor_Num[4],a);digitalWrite(Motor_Num[5],b);break;
    case 4:digitalWrite(Motor_Num[6],a);digitalWrite(Motor_Num[7],b);break;
  }
  return true;
}


uint8_t Obtener_Respuesta(){
  uint8_t response;
      
      Wire.requestFrom(I2C_SLAVE_ADDR, sizeof(response));
      long t = millis();
      while(Wire.available() < 1){
        long delta = millis()-t;
        if(delta>500){
          return 0;
        }
      }
      response =Wire.read();
      //Serial.println("res "+String(response)); 
      return response;
}
void Send_To_Slave(uint8_t data){
  //Serial.println("Enviando : "+String(data));
  Wire.beginTransmission(I2C_SLAVE_ADDR);
  Wire.write((byte*)&data, sizeof(data));
  Wire.endTransmission();
}
