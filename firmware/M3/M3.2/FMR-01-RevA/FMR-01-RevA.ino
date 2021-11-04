//Version 1.1

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
#include "Wire.h" 

bool Sensores_Conectados[4]={1,1,1,1};

int Sensor_Pines[4][3]={
  {4,5,6},//adelante
  {11,12,13},//atras
  {14,15,7},//izquierda
  {16,17,8}//derecha

};

void setup() {
 	//CODIGO esclavo
  	Serial.begin(9600);
  for(int i = 0;i<(4);i++){
    pinMode(Sensor_Pines[i][0],OUTPUT);
    pinMode(Sensor_Pines[i][1],INPUT);
    pinMode(Sensor_Pines[i][2],OUTPUT);
    digitalWrite(Sensor_Pines[i][2],1);
  }//TestearSensores();
  Wire.begin(0x01); //identifico como esclavo 1
  Wire.onReceive(receiveEvent); //Declaro Evento
  Wire.onRequest(Peticion); //Declaro Evento
  Serial.println("Listo");
    
}
byte ProximaSolicitudSensor=0;
byte ProximaSolicitudDistanciaMinima=0;

void loop() {
  TestearSensores();
}
byte AnalizarSensor(int s,int distancia_Min){
  if(s!=7){
    if(Sensores_Conectados[s]){
      
      byte sensor =0.01723 *Analizar(Sensor_Pines[s][0],Sensor_Pines[s][1]);
      //Serial.println("ND"+String(sensor));
      if((sensor>distancia_Min)){
        return 1;
      }else{ 
        return 0;
      }
     
    }else{
      return 0;
    }
  }else{
    return 1;
  }
}
bool TestearSensores(){
  for(int i = 0; i<4;i++){
    //Serial.println();
    //Serial.print("t"+String(i));
    if(1){
      //Si el sensor no devuelve nada lo vuelve a intentar
      
      if(Analizar(Sensor_Pines[i][0],Sensor_Pines[i][1])<=0){
        //Si el sensor no devuelve nada lo vuelve a intentar
        //Serial.print(" F");
        //Serial.println("Intento 1 del senosor "+String(i+1)+" Fallido");
        if(Analizar(Sensor_Pines[i][0],Sensor_Pines[i][1])<=0){
          //Si el sensor no devuelve nada lo vuelve a intentar
          // Serial.print(" F");
          //Serial.println("Intento 2 del senosor "+String(i+1)+" Fallido");
          if(Analizar(Sensor_Pines[i][0],Sensor_Pines[i][1])<=0){
            //Cuando ocurren 3 fallos interpreta que el sensor esta desconectado
            //Si el arduino se esta ejecutando solamente muestra una advertencia en consola
            //Si se esta iniciando no le permite iniciarse
            // Serial.print(" F");
            //Serial.println("Intento 3 del senosor "+String(i+1)+" Fallido");
            
              //Serial.println(Advertencia_Sensor_Desconectado);
              Sensores_Conectados[i] = 0;
              digitalWrite(Sensor_Pines[i][2],0);
            

            }else{
              Sensores_Conectados[i] = 1;
              digitalWrite(Sensor_Pines[i][2],1);
          //Serial.print(" T");  
          }
          }else{
            Sensores_Conectados[i] = 1;
            digitalWrite(Sensor_Pines[i][2],1);
        //Serial.print(" T");  
        }
        }else{
         Sensores_Conectados[i] = 1;
   	     digitalWrite(Sensor_Pines[i][2],1);
        //Serial.print(" T");
      }
    
    }
    
            
  } 
  return true;
}

void receiveEvent(int bytes){
  byte data[bytes];
  for(int i =0;i<bytes;i++){
  	data[i]=Wire.read();
  }
  ProximaSolicitudSensor=data[0]&0b00000111;
  ProximaSolicitudDistanciaMinima=data[0]>>3;
  //Serial.println("Rec "+String(ProximaSolicitudSensor)+" y "+String(ProximaSolicitudDistanciaMinima));
}

void Peticion(){
  
  byte msgRespuesta=AnalizarSensor(ProximaSolicitudSensor,ProximaSolicitudDistanciaMinima);
  //Serial.println("Enviando "+String(msgRespuesta));
  Wire.write(msgRespuesta);
  ProximaSolicitudSensor=0;
  ProximaSolicitudDistanciaMinima=0;
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