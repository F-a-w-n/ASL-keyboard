import cv2
import time
import numpy as np

import keyboard
import mouse
import sys

import mediapipe as mp #getting errors, check version for issues
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

lastTime = int(time.time()*1000)
cap = cv2.VideoCapture(0)
success,image = cap.read()
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = "C:\\Users\\pcklr\\OneDrive\\Documents\\good stuff\\SCHOOL\\gr. 12\\Comp. Eng\\FSE\\exported_model\\gesture_recognizer.task"
base_options = BaseOptions(model_asset_path=model_path)

if len(sys.argv) <= 1:
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Video', cv2.WND_PROP_TOPMOST, 1)
    cv2.resizeWindow('Video', 250, 75)


last_recognized="none recognized"
recognition_time=0
special = {"CURSOR", "CLICK", "SPACE", "ENTER", "none recognized", "Z"}
moveMouse=False
mouseDown=False
oldx, oldy = 1920, 780

# Create a gesture recognizer instance with the live stream mode:
def print_result(result: GestureRecognizerResult, output_image: mp_image, timestamp_ms: lastTime):
    global last_recognized, moveMouse, recognition_time
    if result.gestures and result.gestures[0][0].category_name:
        print('gesture recognition result: {}'.format(result.gestures[0][0].category_name))
        if (result.gestures[0][0].category_name == last_recognized):
            recognition_time += 1
        else:
            recognition_time = 1
        last_recognized='{}'.format(result.gestures[0][0].category_name)
            



options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path="C:\\Users\\pcklr\\OneDrive\\Documents\\good stuff\\SCHOOL\\gr. 12\\Comp. Eng\\FSE\\exported_model\\gesture_recognizer.task"),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result,
    min_tracking_confidence = 0.1,
    min_hand_presence_confidence = 0.1,
    min_hand_detection_confidence = 0.1,
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
            global moveMouse, oldx, oldy
            lmlist = []
            if self.results.multi_hand_landmarks:
                Hand = self.results.multi_hand_landmarks[handNo]
                for id, lm in enumerate(Hand.landmark):
                    h,w,c = image.shape
                    cx,cy = int(lm.x*w), int(lm.y*h)
                    lmlist.append([id,cx,cy])
                    if id == 9:
                        cv2.circle(image,(cx,cy), 15, (255,0,0), cv2.FILLED)
                        if (moveMouse):
                            cx = (w-cx)/w
                            cx = cx-0.1
                            cx *= 1.25 * 1920
                            cx = (cx+oldx)/2
                            cy = cy/h
                            cy = cy-0.2
                            cy *= 1.6 *1080
                            cy = (cy+oldy)/2
                            mouse.move(cx, cy)
                            oldx = cx
                            oldy = cy

            return lmlist

def main():
    global moveMouse, recognition_time, mouseDown
    with GestureRecognizer.create_from_options(options) as recognizer:
        tracker = handTracker()

        while True:
            lastTime = int(time.time()*1000)
            success,image = cap.read()
            outImage = tracker.handsFinder(image)
            tracker.positionFinder(image)

            imageRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imageRGB)
            recognizer.recognize_async(mp_image, lastTime)
            if (recognition_time > 15 and last_recognized == 'CLICK'):
                mouse.press()
                mouseDown = True
            elif mouseDown == True and last_recognized != 'CLICK':
                mouse.release()
                mouseDown = False
            elif (recognition_time == 10):
                moveMouse=False
                if (last_recognized not in special):
                    keyboard.write(last_recognized)
                else:
                    match last_recognized:
                        case 'CURSOR':
                            moveMouse=True
                        case 'CLICK':
                            moveMouse=True
                            mouse.click()
                        case 'ENTER':
                            keyboard.send("enter")
                        case 'SPACE':
                            keyboard.write(" ")
                        case 'Z':
                            keyboard.send("backspace")
                        case "none recognized":
                            pass
                            
            if len(sys.argv) > 1 and sys.argv[1] == "-v":
                outImage = cv2.putText(outImage, last_recognized if last_recognized != 'Z' else 'BACKSPACE', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
                lineWidth  = recognition_time*5 + 50 if (recognition_time < 10) else 100
                colour = (0, 255, 0) if recognition_time >= 10 else (0, 0, 255)
                outImage = cv2.line(outImage, (50, 60), (lineWidth, 60), colour, 5)
                cv2.imshow('Camera Feed', outImage)
            else:
                minImage = np.full((75,250,3), (1,1,1), dtype=np.int8)
                h,w,c = minImage.shape

                textsize = cv2.getTextSize(last_recognized if last_recognized != 'Z' else 'BACKSPACE', cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0]
                minImage = cv2.putText(minImage, last_recognized if last_recognized != 'Z' else 'BACKSPACE', ((250-textsize)//2, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

                lineWidth  = recognition_time*20 + 25  if (recognition_time < 10) else 225
                colour = (0, 255, 0) if recognition_time >= 10 else (0, 0, 255)
                minImage = cv2.line(minImage, (25, 70), (lineWidth, 70), colour, 5)

                cv2.imshow("Video", minImage)
                winx = 0 if (oldx > 2*w or oldy > 2*h) else 1920-625
                cv2.moveWindow('Video', winx, 0)
            
            cv2.waitKey(1);

if __name__ == "__main__":
    main()     