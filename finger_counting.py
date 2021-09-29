import cv2
import time
import os
import handTrackingModule as htm

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

CTime = 0
PTime = 0

tipsId = [4, 8, 12, 16, 20]

folderPath = "fingersImage"
mylist = os.listdir(folderPath)

all_images = []

detector = htm.handDetector()

for name in mylist:
    img = cv2.imread(f"{folderPath}/{name}")
    all_images.append(img)

while True:
    _, frame = cap.read()
    if frame is None:
        break

    img = detector.findHands(frame)

    lmList = detector.findPosition(img, draw= False)
    
    if len(lmList) != 0:
        fingers = []
        for tips in tipsId:
            if tips == 4:
                if lmList[tips][1] > lmList[tips-1][1]:
                    fingers.append(0)
                else:
                    fingers.append(1)
            
            else:
                if lmList[tips][2] < lmList[tips-2][2]:
                    fingers.append(1)
                
                else:
                    fingers.append(0)

        totalfingers = fingers.count(1)  
        
        #frame[0:200, 0:200] = all_images[totalfingers-1]

        cv2.rectangle(frame, (0, 0), (150, 150), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, str(totalfingers), (20, 140), cv2.FONT_HERSHEY_PLAIN,
                     10, (255, 0, 0), 10)
    cv2.imshow("image", frame)
    if cv2.waitKey(1) == ord("q"):
        break
cv2.destroyAllWindows()
cap.release()
