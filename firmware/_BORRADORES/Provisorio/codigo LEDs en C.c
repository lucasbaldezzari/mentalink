/*
 * LEDs.cpp
 *
 * Created: 12/6/2021 19:56:01
 * Author : Emi
 */ 

#define F_CPU 16000000ul
#include <avr/io.h>
#include <util/delay.h>
#include <avr/interrupt.h>

unsigned int contador1 = 0;
unsigned int contador2 = 0;
unsigned int contador3 = 0;
unsigned int contador4 = 0;
unsigned int contador5 = 0;
unsigned int contador6 = 0;


ISR(TIMER1_OVF_vect){
	TCNT1 = 57536; //inicia el conteo para cuando llegue a 2^16 sea 0.5ms
	contador1++; //contador de overflows que ocurren
	if (contador1 == 25){
		PORTC ^= (1<<PORTC0);
		contador1 = 0;
	}
	contador2++; //contador de overflows que ocurren
	if (contador2 == 30){
		PORTC ^= (1<<PORTC1);
		contador2 = 0;
	}
	contador3++; //contador de overflows que ocurren
	if (contador3 == 35){
		PORTC ^= (1<<PORTC2);
		contador3 = 0;
	}
	contador4++; //contador de overflows que ocurren
	if (contador4 == 40){
		PORTC ^= (1<<PORTC3);
		contador4 = 0;
	}
	contador5++; //contador de overflows que ocurren
	if (contador5 == 45){
		PORTC ^= (1<<PORTC4);
		contador5 = 0;
	}
	contador6++; //contador de overflows que ocurren
	if (contador6 == 50){
		PORTC ^= (1<<PORTC5);
		contador6 = 0;
	}
}

int main(void)
{
	DDRC = 0b00111111;
	PINC = 0b00000000;

	cli();
	TIMSK1 |= (1<<TOIE1); //activo por overflow timer 0
	TCCR1B |= (1<<CS10); //pre scaler en 64 2E6/64 es la frecuencia de trabajo
	TCNT1 = 57536; //inicia el conteo para cuando llegue a 2^8 sea 0.5ms (regula el tiempo de muestreo) 2^8-(tiempo*Fcpu/prescaler)
	sei();

	while (1)
	{

	}
}