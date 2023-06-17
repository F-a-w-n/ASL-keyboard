# **ASL Computer Vision Keyboard**

## Description
This application can be used to convert hand signals to keyboard and mouse inputs. 

It uses python with mediapipe and openCV to take camera readings, detect hand landmarks, and provide visual output to users

The model used to detect the various gestures was trained on about 60,000 pictures of hands making the gestures at various angles and positions in 3D space. Given more time for the project, it would be ideal to increase that number and hopefully create a more accurate model. In the repo there is also a folder with old test models trained off smaller datasets through testing.

## How to run
In whichever terminal you use, cd to the project's directory and run
`python main.py`

additionally it can be run with the camera feed for debugging by running
`python main.py -v`

The training script is intended to work in a Google Colab notebook and has not been tested successfully elsewhere.

## Gestures
The Keyboard is based upon American Sign Language gestures for keyboard input. Training a model to effectively recognize more than the stationary alphabetical gestures is far beyond the scope of my current knowledge and resources, so the alphabet is all that is included currently.

Other gestures are limited to left clicking, moving the cursor, space, and enter. Given more time and data, additional gestures would certainly help with things like scrolling, arrows, and numbers. 

Within the repo is an image chart with all the gestures used