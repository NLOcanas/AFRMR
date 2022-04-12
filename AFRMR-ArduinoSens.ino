// Programmer: Nathaniel L Ocanas
// HEC-3 Senior Design: Advanced Fire Rescue Mobile Robot (AFRMR)
// Date:2/20/21

// This code works as an interface for the arduino and sensors for the AFRMR prototype D3
// In this code, the sensors connections are; 2 DS18B20 Temperature Sensor, 
// 1 UART Infrared CO2  CarbonDioxide,2 HC-SR04 Ultrasonic sensor. 
// Part of this code was taken from a public accessible library, The DallasTemperature.h library. 
// This library and code was designed by Rui Santos to work with the DS18B20 sensors
// The code used from his example have been outlined throughout this program.


//RUI SANTOS DALLAS TEMP CODE BEGIN
#include <OneWire.h>
#include <DallasTemperature.h>

// Data wire is connected to the Arduino digital pin 7
#define ONE_WIRE_BUS 7 
// Setup a oneWire instance to communicate with any OneWire devices
OneWire oneWire(ONE_WIRE_BUS);// all Temp data will stream to on pin 7
// Pass our oneWire reference to Dallas Temperature sensor 
DallasTemperature sensors(&oneWire);
int numberofDevices;
DeviceAddress tempDeviceAddress;

int numberOfDevices; // Number of temperature devices found
//RUI SANTOS Dallas Temp sensor CODE END

// Front Back Ultrasonic sensors
// Ultrasonic sensor1
int trigPin1=2;
int echoPin1=3;
// Ultrasonic sensor2
int trigPin2=4;
int echoPin2=5;

void setup() {
  Serial.begin (9600);
// Set pinmode for pins used for Ultrasonic Sensors
  pinMode(trigPin1, OUTPUT);
  pinMode(echoPin1, INPUT);
   pinMode(trigPin2, OUTPUT);
  pinMode(echoPin2, INPUT);

 //Rui Santos Dallas Temp Sensor code  
  sensors.begin();
  // Grab a count of devices on the wire
  numberOfDevices = sensors.getDeviceCount();

  // locate devices on the bus
  Serial.print("Locating devices...");
  Serial.print("Found ");
  Serial.print(numberOfDevices, DEC);
  Serial.println(" devices.");

  // Loop through each device, print out address
  for(int i=0;i<numberOfDevices; i++) {
    // Search the wire for address
    if(sensors.getAddress(tempDeviceAddress, i)) {
      Serial.print("Found device ");
      Serial.print(i, DEC);
      Serial.print(" with address: ");
    //printAddress(tempDeviceAddress,i);
      Serial.println();
    } else {
      Serial.print("Found ghost device at ");
      Serial.print(i, DEC);
      Serial.print(" but could not detect address. Check power and cabling");
    } // end of Rui Santos Dallas Temp Sensor code
  }
}

void loop() {
  delay(500);
  //Dallas Temp Sensor code
  sensors.requestTemperatures(); 
//Serial.print("outside Celsius temperature: ");
//Serial.println(sensors.getTempCByIndex(0));
//Serial.print("internal Celsius temperature: ");
//Serial.println(sensors.getTempCByIndex(1));
  Serial.print(" - Fahrenheit temperature: ");
  Serial.println(sensors.getTempFByIndex(0));
  Serial.print(" - Fahrenheit temperature: ");
  Serial.println(sensors.getTempFByIndex(1));
 //End Dallas Temp Sense code

// set up Ultrasonic sensors 1 & 2  
////////////////////////////////////
// Set up Ultrasonic Sensor1 trigger
  long duration1, distance1;
  digitalWrite(trigPin1, LOW); // Added this line
  delayMicroseconds(2); // Added this line
  digitalWrite(trigPin1, HIGH);
  delayMicroseconds(10); // Added this line
  digitalWrite(trigPin1, LOW);
  duration1 = pulseIn(echoPin1, HIGH);
  distance1 = (duration1/2) / 29.1;
// Iterate through Ultrasonic Sensor1 Echo
   if (distance1 >= 500 || distance1 <= 0){
    //Serial.println("Out of range");
    Serial.println("s1OOR");
  }
  else {
    if(distance1<20){
    //Serial.print("FRONT BLOCKED!\n");
    Serial.print("");
      }
    else{
        //Serial.print("ULTCLEAR\n");
        Serial.print("");
      }    
    Serial.print ( "Sensor1  ");
    Serial.print ( distance1);
    Serial.println("cm");
  }
  delay(2000);
// Set up Ultrasonic Sensor2 trigger
long duration2, distance2;
  digitalWrite(trigPin2, LOW);  // Added this line
  delayMicroseconds(2); // Added this line
  digitalWrite(trigPin2, HIGH);
  delayMicroseconds(10); // Added this line
  digitalWrite(trigPin2, LOW);
  duration2 = pulseIn(echoPin2, HIGH);
  distance2= (duration2/2) / 29.1;
// Iterate through Ultrasonic Sensor2 Echo
   if (distance2 >= 500 || distance2 <= 0){
        //Serial.println("Out of range");
    Serial.println("S2OOR");
  }
  else {
    if(distance2<20){
      //Serial.print("BACK BLOCKED!\n");
       Serial.print("");
       }
    else{
        //Serial.print("CLEAR\n");
         Serial.print("");
        }
    Serial.print("Sensor2  ");
    Serial.print(distance2);
    Serial.println("cm\n");
  }
  delay(500);}
  // Here will be the code for the CO2 infrared sensor. still waiting on device to come in t
  // to test and create practical code for our application.

}
