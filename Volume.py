import cv2 as cv
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.hands as hands
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
import numpy as np
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


class volm:
    def __init__(self):
        # Initialize audio control
        self.defSpeaker = AudioUtilities.GetSpeakers() # this part basically uses your default speaker
        self.interface = self.defSpeaker.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = cast(self.interface, POINTER(IAudioEndpointVolume))

        self.mp_hands = hands
        self.hands_detector = self.mp_hands.Hands()

        # Open the webcam
        self.cap = cv.VideoCapture(0)

        if self.volume is None:
            print("Failed to get volume control interface")
        else:
            print("Volume control interface obtained successfully")
            self.minvol, self.maxvol,_ = self.volume.GetVolumeRange()
            print(f"Volume range: {self.minvol} to {self.maxvol}")

    def process(self):

        while True:
            is_true, frame = self.cap.read()
            if not is_true:
                break

            img2rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB) # mediapipe uses rbg images
            results = self.hands_detector.process(img2rgb) 
            h, w, _ = frame.shape

            lmlist = []
            if results.multi_hand_landmarks: # this is a list of hands,where each hand is a list of normalized landmark eg x,y,z which ranges from 0 to 1
                onehand = results.multi_hand_landmarks[0]
                for id, landmark in enumerate(onehand.landmark):

                    mp_drawing.draw_landmarks(frame, onehand, self.mp_hands.HAND_CONNECTIONS)
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    lmlist.append([id, x, y])

        
                x1, y1 = lmlist[4][1], lmlist[4][2]  # Thumb tip coordinates
                x2, y2 = lmlist[8][1], lmlist[8][2]  # Index finger tip coordinates

                cv.circle(frame, (x1, y1), 10, (0, 255, 0), -1)
                cv.circle(frame, (x2, y2), 10, (0, 255, 0), -1)

                cv.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 2), 10, (0, 255, 0), -1)
                dist = hypot(x1 - x2, y1 - y2)

                cv.line(frame, (x1, y1), (x2, y2), (10, 255, 10), 3)
                if dist < 50:
                    cv.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 2), 10, (90, 105, 110), -1)

                maxdist, mindist = 180.0, 17.0
                inter = np.interp(dist, [mindist, maxdist], [self.minvol, self.maxvol])
                print(f"Distance: {dist}, Interpolated volume: {inter}")

                current_volume = self.volume.GetMasterVolumeLevel()
                if abs(current_volume - inter) > 1.0:  # Adjust threshold as necessary
                    self.volume.SetMasterVolumeLevel(inter, None)
                    print(f"Volume set to: {inter}")

            cv.imshow("Frame", frame)
            if cv.waitKey(1) & 0xFF == ord('x'):
                break

        self.cap.release()
        cv.destroyAllWindows()

vomlObj = volm()
vomlObj.process()
