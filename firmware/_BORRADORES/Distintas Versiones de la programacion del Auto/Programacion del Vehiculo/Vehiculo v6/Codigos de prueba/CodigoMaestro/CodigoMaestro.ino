  //Version 1.0
  
  /*
   * To change this license header, choose License Headers in Project Properties.
   * To change this template file, choose Tools | Templates
   * and open the template in the editor.
   */
  /*--------------------------------------
  Auto Creado por Integrantes de Mentalink
  ---------------------------------------*/
  //Librería Wire
  #include "Wire.h"
  //Lib bt 
  #include <SoftwareSerial.h>
  SoftwareSerial BTSerial(12,13);// Definimos los pines RX y TX
  /*Configuracion Modulo BT
  AT
  AT+NAME=Mentalink
  AT+PSWD=0001
  */
  const int Key_Cambiar_Modo = 200;
  const int Key_Confirmar_Modo = 211;
  const int Key_Agregar_Modo = 222;
  const int Key_Confirmar_Variable=233;
  
  bool Debug_mode = true;//unused var
  
  
  ///////////////////////////////////////////////////
  
  byte Stimulo = 0;
  byte Orden = 0;
  byte Session = 0;
  bool Algo_Adelante=0;
  bool Algo_Atras=0;
  bool Se_Esta_Moviendo=0;
  
  ///////////////////////////////////////////////////
  //                                    //m1 m1 m2 m2 m3 m3 m4 m4// SlaveMode Duracion Active break_val
  
  String Codigos_de_Movimiento_Adelante[12]={"t","f","t","f","f","t","f","t","4","5000","","20"},
    	   Codigos_de_Movimiento_Atras[12]={"f","t","f","t","t","f","t","f","8","5000","","20"},
     	   Codigos_de_Movimiento_GiroIZQ[12]={"f","t","f","t","f","t","f","t","4","5000","dev","23"},
         Codigos_de_Movimiento_GiroDER[12]={"t","f","t","f","t","f","t","f","8","5000","dev","23"},
         Codigos_de_Movimiento_DER[12]={"t","f","t","f","f","t","f","t","4","5000","","0"},
         Codigos_de_Movimiento_IZQ[12]={"f","t","f","t","t","f","t","f","8","5000","","0"},
  	     Codigos_de_Movimiento_Null[12]={"f","f","f","f","f","f","f","f","1","10","","1"};
  
  byte Cantidad_de_Modos=7;
  
  int Motor_Num[8]={3,4,5,6,7,8,9,10};
  //                {2,6,11,10,17,16,15,14}
  
  byte acutal_Motor_Modes[4]={0,0,0,0};


  byte I2C_SLAVE_ADDR =0x01;
  
  byte Mode=0;
  
  void setup(){
  
    long t1 = millis();
    	Wire.begin();
      for(int i = 0;i<8;i++){
        pinMode(Motor_Num[i],OUTPUT);
  
      }
      pinMode(11,OUTPUT);
      analogWrite(11,255);
   	  Serial.begin(9600);
    	Serial.setTimeout(10);
      Serial.println("AT NAME?");
    	Serial.println(1);
    	Serial.println("Comunicación abierta;");
  	  BTSerial.begin(9600);
      BTSerial.write("ready!");
          while(true){
              if(Iniciar()){break;}
          }
          Serial.println("Iniciado");
    Serial.print(millis()-t1);Serial.println(" mm");
    
  }
  
  
  
  
        
  void loop() {
    if(BTSerial.available()){
       byte res=(BTSerial.read());
       Control(res);
         
         // 0b0000 00       0/1 0/1

      }
  }
     // Mode = res;
     // CambiarMovimiento(Mode);
     // Serial.println("Recived = "+String(Mode));
  void SendToPC(){
    //0000 0(moviendose)(algo adelante)(algo atras)
    BTSerial.write((byte)(Algo_Adelante<<1)|Algo_Atras|(Se_Esta_Moviendo<<2));
    Algo_Adelante=0;
    Algo_Atras=0;
  }
  
  
  
  
  
  void Control(byte val2){
    byte Session = ((val2)&0b00000001);
    byte Stimulo = ((val2>>1)&0b00000001);
    byte Orden = ((val2>>2)&0b00000111);
    if (Stimulo == 0)
      {
        CambiarMovimiento(Orden);
      }
      else 
      Serial.println("Algo salió mal al recibir un dato de movimiento");
    }    
  
  
  bool Iniciar_Movimiento(int codigo){
    //if(codigo>Cantidad_de_Modos){Mode=0;Serial.println("Error al leer el modo");return false;}
    Serial.print("Iniciando Movimiento : ");Serial.println(codigo);
    	if( CambiarModo(getCodigoDeMovimiento(codigo,8).toInt()) ){
        if(!Motor(1,getCodigoDeMovimiento(codigo,0),getCodigoDeMovimiento(codigo,1))){ApagarTodo();return false;}
    		if(!Motor(2,getCodigoDeMovimiento(codigo,2),getCodigoDeMovimiento(codigo,3))){ApagarTodo();return false;}
    		if(!Motor(3,getCodigoDeMovimiento(codigo,4),getCodigoDeMovimiento(codigo,5))){ApagarTodo();return false;}
    		if(!Motor(4,getCodigoDeMovimiento(codigo,6),getCodigoDeMovimiento(codigo,7))){ApagarTodo();return false;}
        	Mode=codigo;
        	return true;
    	}else{
    		Serial.println("Algo salió mal al comunicarse con el esclavo");
        	return false;
    	}
  }
  byte PedirDatos(){
  	  Send_To_Slave(Key_Confirmar_Variable);
    	return Obtener_Respuesta();
  }
  void CambiarMovimiento(byte Codigo){
    Serial.println("Intentando cambiar movimiento");
    	if(Iniciar_Movimiento(Codigo)){
        long t1 = millis();
        int delta=0;
        while(delta<getCodigoDeMovimiento(Codigo,9).toInt()){
          Se_Esta_Moviendo=1;
          bool brk = false;
          SendToPC();
          switch(Codigo){
            	default:brk=true;break;
            	case 5:case 6:case 1:if(PedirDatos()<getCodigoDeMovimiento(Codigo,11).toInt()){
                Algo_Adelante=1;
          			brk=true;
                
          		}break;
              case 2:if(PedirDatos()<getCodigoDeMovimiento(Codigo,11).toInt()){
                Algo_Atras=1;
          			brk=true;
                
          		}break;
            	case 3:case 4:if(PedirDatos()>getCodigoDeMovimiento(Codigo,11).toInt()){
          			brk=true;
          		}break;
         	}
          if(brk){break;}
          long t2 = millis();
        	delta += t2-t1;
          t1=t2;
        }		
        Se_Esta_Moviendo=0;
        SendToPC();
        Iniciar_Movimiento(0);
    	}
  }
  bool CambiarModo(int Codigo){
  	Send_To_Slave(Key_Confirmar_Modo);
      Send_To_Slave(5);
    	if(Obtener_Respuesta()!=Codigo){
        Serial.println("Codigo de respuesta A");
          Send_To_Slave( Key_Cambiar_Modo);
          Send_To_Slave(Codigo);
        	Send_To_Slave(Key_Confirmar_Modo);
      	Send_To_Slave(5);
        	if(Obtener_Respuesta()!=Codigo){
            Serial.println("Codigo de respuesta B");
          	Send_To_Slave( Key_Cambiar_Modo);
          	Send_To_Slave(Codigo);
        		Send_To_Slave(Key_Confirmar_Modo);
      		  Send_To_Slave(5);
            if(Obtener_Respuesta()!=Codigo){
              Serial.println("Codigo de respuesta C");
          	  Send_To_Slave( Key_Cambiar_Modo);
          	  Send_To_Slave(Codigo);
        		  Send_To_Slave(Key_Confirmar_Modo);
      			  Send_To_Slave(5);
              if(Obtener_Respuesta()==Codigo){return true;}
      		  }else{
      			  return true;
      		  }
      	  }else{
      		  return true;
      	  }
        }else{
      	  return true;
        }
        
    	  return false;
    	
  }
  void ApagarTodo(){
    	for(int i = 0;i<8;i++){
      	digitalWrite(Motor_Num[i],false);
      }
    	for(int i =0;i<4;i++){
      	acutal_Motor_Modes[i]=0;
    	}
    Mode=0;
  }
  bool Motor(int motor,String a,String b){
    if(a=="t" && b=="t"){Serial.println("Error Prendiendo Motor "+String(motor)+"\nLos datos del Modo estan mal ingresados");return false;}
    if(a=="t"){acutal_Motor_Modes[motor]=1;}else
    if(b=="t"){acutal_Motor_Modes[motor]=2;}else
    {acutal_Motor_Modes[motor]=0;}
    switch(motor){
      default:return false;
    	case 1:digitalWrite(Motor_Num[0],a=="t");digitalWrite(Motor_Num[1],b=="t");break;
      case 2:digitalWrite(Motor_Num[2],a=="t");digitalWrite(Motor_Num[3],b=="t");break;
      case 3:digitalWrite(Motor_Num[4],a=="t");digitalWrite(Motor_Num[5],b=="t");break;
      case 4:digitalWrite(Motor_Num[6],a=="t");digitalWrite(Motor_Num[7],b=="t");break;
    
    }
    Serial.println(acutal_Motor_Modes[motor]);
    return true;
  }
  
  String getCodigoDeMovimiento(int codigo,int codigo2){
  //la existencia de este metodo se basa en que los vectores se buguean de forma absurda
  //usando la funcion .length revientan y no se porque
    	switch(codigo){
        	default:return Codigos_de_Movimiento_Null[codigo2];
    		  case 1:return  Codigos_de_Movimiento_Adelante[codigo2];
        	case 2:return  Codigos_de_Movimiento_Atras[codigo2];
        	case 3:return  Codigos_de_Movimiento_GiroIZQ[codigo2];
        	case 4:return  Codigos_de_Movimiento_GiroDER[codigo2];
          case 5:return  Codigos_de_Movimiento_DER[codigo2];
          case 6:return  Codigos_de_Movimiento_IZQ[codigo2];
      }
  
  }
  
  bool Iniciar(){
    	ApagarTodo();
    	
      Send_To_Slave(Key_Confirmar_Modo);
      Send_To_Slave(5);
      if(Obtener_Respuesta()!=1){
        
          Send_To_Slave( Key_Cambiar_Modo);
          Send_To_Slave(1);
      }
      if(!AgregarModo(2,getCodigoDeMovimiento(1,8).toInt(),40,getCodigoDeMovimiento(1,10)=="dev")){return false;}
      if(!AgregarModo(3,getCodigoDeMovimiento(2,8).toInt(),40,getCodigoDeMovimiento(2,10)=="dev")){return false;}
      //if(!AgregarModo(4,getCodigoDeMovimiento(3,8).toInt(),40,getCodigoDeMovimiento(3,10)=="dev")){return false;}
  	return true;
  }
  
  /*void Cambiar_Modo(int valor){
      Send_To_Slave(200);
      Send_To_Slave(getCodigoDeMovimiento(valor)[8].toInt());
  }*/
  
  bool AgregarModo(int pos,int a, int b, int c){
      int delaytime=0;
    	int tiempo_reseteo=1500;
    	Send_To_Slave(Key_Agregar_Modo);
      Send_To_Slave(pos);delay(delaytime);
      Send_To_Slave(a);delay(delaytime);
      Send_To_Slave(b);delay(delaytime);
      Send_To_Slave(c);delay(delaytime);
      Send_To_Slave(Key_Confirmar_Modo);delay(delaytime);
      Send_To_Slave(a);delay(delaytime);
      byte r1=Obtener_Respuesta();delay(delaytime);
    	if(r1==0){
          Send_To_Slave(Key_Agregar_Modo);
          Send_To_Slave(pos);delay(delaytime);
          Send_To_Slave(a);delay(delaytime);
          Send_To_Slave(b);delay(delaytime);
          Send_To_Slave(c);delay(delaytime);
          Send_To_Slave(Key_Confirmar_Modo);delay(delaytime);
          Send_To_Slave(a);delay(delaytime);
          r1=Obtener_Respuesta();
        	
          if(r1==0){
  			Serial.println("Esta saltando un error cuando se solicitan los datos");
              Send_To_Slave( Key_Cambiar_Modo);
              Send_To_Slave(0);
              delay(tiempo_reseteo);
              return false;
          }
      }
    	byte r2=Obtener_Respuesta();delay(delaytime);
      byte r3=Obtener_Respuesta();delay(delaytime);
    Serial.println(("Recivido de la confirmacion de modo ")+String(r1)+(" ")+String(r2)+(" ")+String(r3)+(" "));
      if(!(r1==a&&r3==c)){
          Send_To_Slave(Key_Agregar_Modo);
          Send_To_Slave(pos);delay(delaytime);
          Send_To_Slave(a);delay(delaytime);
          Send_To_Slave(b);delay(delaytime);
          Send_To_Slave(c);delay(delaytime);
          Send_To_Slave(Key_Confirmar_Modo);delay(delaytime);
          Send_To_Slave(a);delay(delaytime);
          int r1=Obtener_Respuesta();
          if(r1==0){
              Send_To_Slave(Key_Agregar_Modo);
              Send_To_Slave(pos);delay(delaytime);
              Send_To_Slave(a);delay(delaytime);
              Send_To_Slave(b);delay(delaytime);
              Send_To_Slave(c);delay(delaytime);
              Send_To_Slave(Key_Confirmar_Modo);delay(delaytime);
              Send_To_Slave(pos);delay(delaytime);
              r1=Obtener_Respuesta();
              if(r1==0){
            		Serial.print(pos);
                  Send_To_Slave( Key_Cambiar_Modo);
                  Send_To_Slave(0);
                  delay(tiempo_reseteo);
                  return false;
              }
          }
          int r2=Obtener_Respuesta();
          int r3=Obtener_Respuesta();
          if(!(r1==a&&r2==b&&r3==c)){
              Send_To_Slave( Key_Cambiar_Modo);
              Send_To_Slave(0);
              delay(tiempo_reseteo);
              return false;
          }
      }else{
        Serial.println("    Se Añadio Correctamente");
          return true;
      }
  }
  
  byte Obtener_Respuesta(){
  	byte response = 0;
     	Wire.requestFrom(I2C_SLAVE_ADDR, sizeof(response));
     	uint8_t index = 0;
     	byte* pointer = (byte*)&response;
    	while(true){
        if(Wire.available()){
     		while (Wire.available())
     		{
        		*(pointer + index) = (byte)Wire.read();
        		index++;
     		}
          
        }{break;}
    	}
    	return response;
  }
  void Send_To_Slave(byte data){
   	  Wire.beginTransmission(I2C_SLAVE_ADDR);
     	Wire.write((byte*)&data, sizeof(data));
     	Wire.endTransmission();
      Serial.println("Se envio : "+String(data));
  }
