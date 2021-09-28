#include <SoftwareSerial.h>  // 

SoftwareSerial BT(10, 11);  // pin 10 TX, pin 11 RX

char Dt = 0; 
char val = 0;

byte Stimulo = 0;
byte Orden = 0;
byte Session = 0;

char testled = 2;

volatile unsigned int cuenta = 0;
bool estado = false;

byte mascaraComando = 0b00000111;

unsigned int acumulador = 0; 
   
int Detenido = 2;
int Adelante = 3;
int Atras = 4;
int Derecha = 5;
int Izquierda = 6;
int RotD = 7;
int RotI = 8;

void setup(){
  pinMode(testled,OUTPUT);
  BT.begin(9600);    
  
// iniTimer0();
  
  pinMode(Detenido,OUTPUT);
  pinMode(Adelante,OUTPUT);
  pinMode(Atras,OUTPUT);
  pinMode(Derecha,OUTPUT);
  pinMode(Izquierda,OUTPUT);
  pinMode(RotD,OUTPUT);
  pinMode(RotI,OUTPUT);
}

void loop()
{
  if (BT.available())
  {     
  Dt = BT.read();   
  Control(Dt);
}
}

void Movimiento(byte val)
{
  switch(val)
  {
    case 0: 
      digitalWrite(Detenido, 1);
      break;
    case 1: 
    digitalWrite(Adelante, 1);
    digitalWrite(Detenido, 0);
      break;
    case 2: 
      digitalWrite(Atras, 1);
      digitalWrite(Detenido, 0);
      break;
    case 3: 
      digitalWrite(Derecha, 1);
      digitalWrite(Detenido, 0);
      break;
    case 4: 
      digitalWrite(Izquierda, 1);
      digitalWrite(Detenido, 0);
      break;
    case 5: 
      digitalWrite(RotD, 1);
      digitalWrite(Detenido, 0);
      break;
    case 6: 
      digitalWrite(RotI, 1);
      digitalWrite(Detenido, 0);
      break;
  }
};

void Control(byte val2)
{
  byte Session = ((val2)&0b00000001);
  byte Stimulo = ((val2>>1)&0b00000001);
  byte Orden = ((val2>>2)&0b00000111);
  if (Stimulo == 0)
    {
      Movimiento(Orden);
    }
    else Stop();
  backComand();
  }

void Stop()
{
  digitalWrite(Detenido, 1);
  digitalWrite(Adelante, 0);
  digitalWrite(Atras, 0);
  digitalWrite(Derecha, 0);
  digitalWrite(Izquierda, 0);
  digitalWrite(RotD, 0);
  digitalWrite(RotI, 0);
}

void backComand()
{
  BT.write(0b00100000);
  }

void iniTimer0()
{
//Seteamos el Timer0 para que trabaje a 5000Hz = 0.2ms
  TCCR0A = 0;// pongo a cero el registro de control del timer1
  TCCR0B = 0;// Lo mismo para el TCCR0B
  TCNT0  = 0;//initialize counter value to 0
  
  // turn on CTC mode
  TCCR0A |= (1 << WGM01);//Ponemos un 1 en el Bit WGM01 del registro TCCR0A - Modo CTC (ver página 107)
  // Seteamos el PreScaler en 64 (ver página 109 de la hoja de datos)
  TCCR0B |= (0 << CS02) | (0 << CS01) | (1 << CS00);
  //int preScaler = 64UL;
  // Cargamos el comparador del Timer0 para que nos de una interrupción aproximadamente de 0.1ms
  //unsigned char comparador = ((F_CPU/(PRE_SCALER*frecTimer)) - 1);
  OCR0A = 159;
  //OCR0A = 49;// = (16MHz/(preScaler*frecuencia de Interrupción))-1

  //Habilito la interrupción (ver pagina 110 de hoja de datos)
  TIMSK0 |= (1 << OCIE0A);
  }
//
// ISR(TIMER0_COMPA_vect)//Rutina interrupción Timer0.
//{
//  estado = !estado;
//  digitalWrite(13,estado);
//};
