/*
  This code uses a neural network  to recognize digits using the tiny machine learning kit. 

 The neural network is created and trained using this link: https://colab.research.google.com/drive/1oqLV6Uvoo6HllVIfYQkYlJT-cWMVZ27L?usp=sharing
 The trained model needs to be added as a model.h file next to this file. 
  Hardware: Arduino Nano 33 BLE Sense board.


Based on an example made by
  Created by Don Coleman, Sandeep Mistry
  Adapted by Dominic Pajak
  This example code is in the public domain.
  Adapted to a camera digit recognizer application by Bart Bozon

Bart Bozon @www.bozon.org or https://www.youtube.com/@bartbozon

Install the Harvard_TinyMLx library to work!!!

*/
#include <TinyMLShield.h>

// Arduino_TensorFlowLite - Version: 0.alpha.precompiled
#include <TensorFlowLite.h>

#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>
#include "model.h"

// global variables used for TensorFlow Lite (Micro)
tflite::MicroErrorReporter tflErrorReporter;

// pull in all the TFLM ops, you can remove this line and
// only pull in the TFLM ops you need, if would like to reduce
// the compiled size of the sketch.
tflite::AllOpsResolver tflOpsResolver;

const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

// Create a static memory buffer for TFLM, the size may need to
// be adjusted based on the model you are using
constexpr int tensorArenaSize = 8 * 1024;
byte tensorArena[tensorArenaSize];

// array to map outcometo a name
const char* CLASSES[] = {
  "0",
  "1",
  "2",
  "3",
  "4",
  "5",
  "6",
  "7",
  "8",
  "9"
};

#define NUM_CLASSES (sizeof(CLASSES) / sizeof(CLASSES[0]))

bool commandRecv = false;    // flag used for indicating receipt of commands from serial port
bool logging = false;        // flag used to print out the raw values.
byte image[176 * 144 * 2];   // QCIF: 176x144 x 2 bytes per pixel (RGB565)
float image_small[88 * 72];  // QCIF: 176x144 x 2 bytes per pixel (RGB565)
float ultra_small[44 * 36];  // QCIF: 176x144 x 2 bytes per pixel (RGB565)
float cut_out[28 * 28];
int bytesPerFrame;
int a;
int16_t pixel;
int16_t red;
int16_t green;
int16_t blue;
float grayscale;

void setup() {
  Serial.begin(9600);
  while (!Serial) {};

  Serial.println("Digit recognizition using the tiny machine learning kit");
  Serial.println("Used camera: OV7675");
  Serial.println("Arduino Nano 33 BLE Sense running TensorFlow Lite Micro");
  Serial.println("");

  // get the TFL representation of the model byte array
  tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1)
      ;
  }

  // Create an interpreter to run the model
  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize, &tflErrorReporter);

  // Allocate memory for the model's input and output tensors
  tflInterpreter->AllocateTensors();

  // Get pointers for the model's input and output tensors
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);

  // Initialize the OV7675 camera
  if (!Camera.begin(QCIF, RGB565, 1, OV7675)) {
    Serial.println("Failed to initialize camera");
    while (1)
      ;
  }
  bytesPerFrame = Camera.width() * Camera.height() * Camera.bytesPerPixel();

  Serial.println("Welcome to the OV7675 test\n");
  Serial.println("press return to start a capture & recognition");
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      commandRecv = true;
    }
  }

  if (commandRecv) {
    commandRecv = false;
    Camera.readFrame(image);
    // empty buffer
    for (int i = 0; i < 88; i++) {
      for (int j = 0; j < 72; j++) {
        image_small[j * 88 + i] = 0;
      }
    }
    // empty buffer
    for (int i = 0; i < 44; i++) {
      for (int j = 0; j < 36; j++) {
        ultra_small[j * 44 + i] = 0;
      }
    }
    // downsize
    for (int i = 0; i < 176; i++) {
      for (int j = 0; j < 144; j++) {
        // a = (int)((image[176 * j * 2 + (175 - i) * 2] * 255 + image[176 * j * 2 + (175 - i) * 2 + 1]) / 7000);
        // a = (a + abs(a)) / 2;
        pixel = image[176 * j * 2 + (175 - i) * 2] * 255 + image[176 * j * 2 + (175 - i) * 2 + 1];
        red = ((pixel & 0xF800) >> 11);
        green = ((pixel & 0x07E0) >> 5);
        blue = (pixel & 0x001F);
        grayscale = (0.2126 * red) + (0.7152 * green / 2.0) + (0.0722 * blue);
        image_small[int((j / 2) * 88 + i / 2)] += grayscale;
        if (logging) { Serial.print(a); }
      }
      if (logging) { Serial.println(); }
    }
    // downsize
    for (int i = 0; i < 88; i++) {
      for (int j = 0; j < 72; j++) {
        ultra_small[int((j / 2) * 44 + i / 2)] += image_small[j * 88 + i];
      }
    }
    Serial.println("Determining max and min pixel values");
    float max = 0;
    float min = 100000;
    for (int i = 0; i < 44; i++) {
      for (int j = 0; j < 36; j++) {
        if (ultra_small[j * 44 + i] > max) {
          max = ultra_small[j * 44 + i];
        }
        if (ultra_small[j * 44 + i] < min) {
          min = ultra_small[j * 44 + i];
        }
      }
    }
    Serial.println(min);
    Serial.println(max);

    for (int i = 0; i < 44; i++) {
      for (int j = 0; j < 36; j++) {
        ultra_small[j * 44 + i] = 1 - (1 * (ultra_small[j * 44 + i] - min) / ((max - min) * 1.0));
      }
    }
    int k = 0;
    // final cut needs to be square
    for (int i = 10; i < 38; i++) {
      for (int j = 5; j < 33; j++) {
        cut_out[k] = ultra_small[j * 44 + i];
        k = k + 1;
      }
    }

    Serial.println("Final result, the input for the NN model");
    for (int i = 0; i < 28; i++) {
      for (int j = 0; j < 28; j++) {
        if (int(9 * (cut_out[i * 28 + j])) < 1) {
          Serial.print(" ");
        } else if (int(9 * (cut_out[i * 28 + j])) < 2) {
          Serial.print(".");
        } else if (int(9 * (cut_out[i * 28 + j])) < 3) {
          Serial.print(":");
        } else if (int(9 * (cut_out[i * 28 + j])) < 4) {
          Serial.print("-");
        } else if (int(9 * (cut_out[i * 28 + j])) < 5) {
          Serial.print("=");
        } else if (int(9 * (cut_out[i * 28 + j])) < 6) {
          Serial.print("+");
        } else if (int(9 * (cut_out[i * 28 + j])) < 7) {
          Serial.print("*");
        } else if (int(9 * (cut_out[i * 28 + j])) < 8) {
          Serial.print("#");
        } else if (int(9 * (cut_out[i * 28 + j])) < 9) {
          Serial.print("%");
        } else {
          Serial.print("@");
        }
      }
      Serial.println("|");
    }

    // input sensor data to model

    for (int i; i < 784; i++) {
      tflInputTensor->data.f[i] = cut_out[i];
    }

    // Run inferencing
    TfLiteStatus invokeStatus = tflInterpreter->Invoke();
    if (invokeStatus != kTfLiteOk) {
      Serial.println("Invoke failed!");
      while (1)
        ;
      return;
    }

    // Output results
    for (int i = 0; i < NUM_CLASSES; i++) {
      Serial.print(CLASSES[i]);
      Serial.print(" ");
      Serial.print(int(tflOutputTensor->data.f[i] * 100));
      Serial.print("%\n");
    }

    if (logging) {
      Serial.println("=============================================================");
      for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
          //Serial.print(grey_values[cut_out[(j - 5) * 28 + (i - 10)]]);
          Serial.print(cut_out[i * 28 + j]);
          Serial.print(",");
        }
        Serial.println();
      }
      Serial.println("outcome");
      for (int i = 0; i < NUM_CLASSES; i++) {
        Serial.print(int(tflOutputTensor->data.f[i] * 100));
        Serial.print(",");
      }
      Serial.println();
    }
    Serial.println("Ready, press return for a new measurement");
  }
}