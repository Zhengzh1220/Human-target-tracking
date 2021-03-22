import cv2
import sys

capture = cv2.VideoCapture(1)
capture.set(3, 640)  
capture.set(4, 480)
tracker = cv2.TrackerKCF_create()


ret, frame = capture.read()
if not ret:
    print ("Cannot read video file")
    sys.exit()

while(True):
    ret, frame = capture.read()
    p1 = (240,120)
    p2 = (400,440)
    cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
    cv2.imshow("Tracking", frame)

    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

cv2.waitKey(2000)
#bbox = cv2.selectROI(frame, False)
bbox = (240,120,160,320)

lx = bbox[2]
ly = bbox[3]

dx = int(ly/10)

cx = int((640-lx)/ dx)
cy = int((480-ly)/ dx)

for i in range(30):
    ret, frame = capture.read()
    p1 = (240,120)
    p2 = (400,440)
    cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
    cv2.imshow("Tracking", frame)

ret = tracker.init(frame, bbox)
dirName = "./mydata/"

for i in range(cx*cy*3):
    ret, frame = capture.read()
    if not ret:
        break

    timer = cv2.getTickCount()
    ret, bbox = tracker.update(frame)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if ret:
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))

        imgPos = frame[p1[1]:p2[1], p1[0]:p2[0]]
        imgPos = cv2.resize(imgPos,(64, 128))
        posName = dirName + "pos/" + str(i) + ".png"
        cv2.imwrite(posName, imgPos)
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

    else:
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Display tracker type on frame
    cv2.putText(frame, "KCF Tracker :  "+ str(cx) + str(cy), (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

    # Display FPS on frame
    cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

    # Display result
    cv2.imshow("Tracking", frame)

    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

count = 0
while(True):
    ret, frame = capture.read()

    k = cv2.waitKey(1) & 0xff
    # if k == ord('q'):
    #
    #     for i in range(cx):
    #         for j in range(cy):
    #             count+=1
    #             imgNeg = frame[j*dx:j*dx+ly, i*dx:i*dx+lx]
    #             imgNeg = cv2.resize(imgNeg, (64, 128))
    #             negName = dirName + "neg/" + str(count) + ".png"
    #             cv2.imwrite(negName, imgNeg)

    cv2.putText(frame, "finsh pos img", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    cv2.imshow("Tracking", frame)

    if k == 27:
        break


cv2.destroyAllWindows()

