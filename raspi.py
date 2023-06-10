import cv2
import time
import numpy as np

from picamera.array import PiRGBArray
from picamera import PiCamera

import keyboard
import mouse

import mediapipe as mp #getting errors, check version for issues
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

camera = PiCamera()
camera.resolution = (1280, 720)
camera.framerate = 32

time.sleep(0.1)
rawCapture = PiRGBArray(camera, size=(1280, 720))

lastTime = int(time.time()*1000)
camera.capture(rawCapture, format="bgr")
image = rawCapture.array
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = "C:\\Users\\pcklr\\OneDrive\\Documents\\good stuff\\SCHOOL\\gr. 12\\Comp. Eng\\FSE\\exported_model\\gesture_recognizer.task"
base_options = BaseOptions(model_asset_path=model_path)

last_recognized="none recognized"
recognition_time=time.time()
special = {"CURSOR", "CLICK", "SPACE", "ENTER"}
moveMouse=False

# Create a gesture recognizer instance with the live stream mode:
def print_result(result: GestureRecognizerResult, output_image: mp_image, timestamp_ms: lastTime):
    global last_recognized, moveMouse, recognition_time
    if result.gestures and result.gestures[0][0].category_name:
        print('gesture recognition result: {}'.format(result.gestures[0][0].category_name))
        if (result.gestures[0][0].category_name != last_recognized):
            last_recognized='{}'.format(result.gestures[0][0].category_name)
            recognition_time = time.time()
            



options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path="C:\\Users\\pcklr\\OneDrive\\Documents\\good stuff\\SCHOOL\\gr. 12\\Comp. Eng\\FSE\\exported_model\\gesture_recognizer.task"),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result,
    min_tracking_confidence = 0.125,
    min_hand_presence_confidence = 0.125,
    min_hand_detection_confidence = 0.125,
    num_hands = 1)
#recognizer = GestureRecognizer.create_from_options(options)




class handTracker():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5,modelComplexity=1,trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def handsFinder(self,image,draw=True):
            imageRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            self.results = self.hands.process(imageRGB)
            
            if self.results.multi_hand_landmarks:
                for handLms in self.results.multi_hand_landmarks:

                    if draw:
                        self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)
            return image

    def positionFinder(self,image, handNo=0):
            lmlist = []
            if self.results.multi_hand_landmarks:
                Hand = self.results.multi_hand_landmarks[handNo]
                for id, lm in enumerate(Hand.landmark):
                    h,w,c = image.shape
                    cx,cy = int(lm.x*w), int(lm.y*h)
                    lmlist.append([id,cx,cy])
                    if id == 9:
                        cv2.circle(image,(cx,cy), 15 , (0,0,255), cv2.FILLED)
                        if (moveMouse):
                            mouse.move((w-cx)/w*1920, cy/h*1080)

            return lmlist

def main():
    global moveMouse, recognition_time
    with GestureRecognizer.create_from_options(options) as recognizer:
        tracker = handTracker()

        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            lastTime = int(time.time()*1000)
            image = frame.array
            outImage = tracker.handsFinder(image)
            lmList = tracker.positionFinder(image)
            
            imageRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imageRGB)
            recognizer.recognize_async(mp_image, lastTime)
            #if len(lmList) != 0:
            #    print(lmList[4])
            if (time.time() - recognition_time > 0.5):
                moveMouse=False
                if (last_recognized not in special):
                    keyboard.write(last_recognized)
                else:
                    match last_recognized:
                        case 'CURSOR':
                            moveMouse=True
                        case 'CLICK':
                            mouse.click()
                        case 'ENTER':
                            keyboard.send("enter")
                        case 'SPACE':
                            keyboard.write(" ")
                        case "none recognized":
                            pass
    
            outImage = cv2.putText(outImage, last_recognized, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

            cv2.imshow("Video",outImage)
            key = cv2.waitKey(1) & 0xFF
            rawCapture.truncate(0)
            if key == ord("q"):
              break

if __name__ == "__main__":
    main()     