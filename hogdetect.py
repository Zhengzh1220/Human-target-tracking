import cv2
import numpy as np
import random
import os

def loadPosImg(filepath):
    imageList = []
    pathDir = os.listdir(filepath)
    for allDir in pathDir:
        child = os.path.join(filepath, allDir)
        if os.path.isfile(child):
            image = cv2.imread(child)
            imageList.append(image)

    return imageList

def loadImageList(dirName, fileListPath):
    imageList = []
    subList = []
    file = open(dirName + '/' + fileListPath)
    imageName = file.readline()

    while imageName != '':
        imageName = dirName + '/' + imageName.split('/', 1)[1].strip('\n')
        #print (imageName)
        imageList.append(cv2.imread(imageName))
        imageName = file.readline()

    for i in range(200):
        n = random.randint(1, len(imageList)-1)
        subList.append(imageList[n])

    return subList

def getPosSample(imageList):
    posList = []
    for i in range(len(imageList)):
        roi = imageList[i]
        posList.append(roi)
    return posList


def getNegSample(imageList):
    negList = []
    random.seed(1)
    for i in range(len(imageList)):
        for j in range(10):
            y = int(random.random() * (len(imageList[i]) - 128))
            x = int(random.random() * (len(imageList[i][0]) - 64))
            negList.append(imageList[i][y:y + 128, x:x + 64])
    return negList


def getHOGList(imageList):
    HOGList = []
    hog = cv2.HOGDescriptor()
    for i in range(len(imageList)):
        gray = cv2.cvtColor(imageList[i], cv2.COLOR_BGR2GRAY)
        HOGList.append(hog.compute(gray))
    return HOGList


def getHOGDetector(svm):
    sv = svm.getSupportVectors()
    rho, _, _ = svm.getDecisionFunction(0)
    sv = np.transpose(sv)
    return np.append(sv, [[-rho]], 0)

def getHardExamples(negImageList, svm):
    hardNegList = []
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(getHOGDetector(svm))
    for i in range(len(negImageList)):
        rects, wei = hog.detectMultiScale(negImageList[i], winStride=(4, 4), padding=(8, 8), scale=1.05)
        for (x, y, w, h) in rects:
            hardExample = negImageList[i][y:y + h, x:x + w]
            hardNegList.append(cv2.resize(hardExample, (64, 128)))
    return hardNegList

def fastNonMaxSuppression(boxes, sc, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = sc
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the score of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick]


labels = []
posImageList = []
posList = []

hosList = []
tem = []
hardNegList = []


posImageList = loadPosImg("./mydata/pos")
print ("posImageList:", len(posImageList))
posList = getPosSample(posImageList)
print ("posList", len(posList))

hosList = getHOGList(posList)
print ("hosList", len(hosList))

[labels.append(+1) for _ in range(len(posList))]


mynegImageList = loadPosImg("./mydata/neg")
print("mynegImageList:", len(mynegImageList))
mynegList = getPosSample(mynegImageList)
print("mynegList: ", len(mynegList))


negImageList = loadImageList("./INRIAPerson/train_64x128_H96", "neg.lst")
print ("offical negImageList:", len(negImageList))
negList = getNegSample(negImageList)
print ("offical negList:", len(negList))

negImageList.extend(mynegImageList)
print ("negImageList:", len(negImageList))
negList.extend(mynegList)
print ("negList:", len(negList))

hosList.extend(getHOGList(negList))
print ("hosList", len(hosList), type(hosList),type(np.array(hosList)))

[labels.append(-1) for _ in range(len(negList))]
print ("labels", len(labels), type(labels),type(np.array(labels)))


svm = cv2.ml.SVM_create()
svm.setCoef0(0.0)
svm.setDegree(3)
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000, 1e-3)  #end param
svm.setTermCriteria(criteria)
svm.setGamma(0)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setNu(0.5)
svm.setP(0.1)  # for EPSILON_SVR, epsilon in loss function?
svm.setC(0.01)  # From paper, soft classifier
svm.setType(cv2.ml.SVM_EPS_SVR)  # C_SVC # EPSILON_SVR # may be also NU_SVR # do regression task
svm.train(np.array(hosList), cv2.ml.ROW_SAMPLE, np.array(labels))


hardNegList = getHardExamples(negImageList, svm)
hosList.extend(getHOGList(hardNegList))
print ("hosList=====", len(hosList))
[labels.append(-1) for _ in range(len(hardNegList))]

# add hard example
svm.train(np.array(hosList), cv2.ml.ROW_SAMPLE, np.array(labels))

# save model
hog = cv2.HOGDescriptor()
hog.setSVMDetector(getHOGDetector(svm))
hog.save('myHogDector.bin')

# person detect

print("finish training")
hog = cv2.HOGDescriptor()
hog.load('myHogDector.bin')


capture = cv2.VideoCapture(1)
capture.set(3,640) 
capture.set(4,480)
tracker = cv2.TrackerKCF_create()

ret, frame = capture.read()
if not ret:
    print ("Cannot read video file")
    sys.exit()

while(True):

    ret, frame = capture.read()
    frame = cv2.resize(frame,(320, 240))
    if not ret:
        break

    timer = cv2.getTickCount()

    rects, scores = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)

    # fastNonMaxSuppression
    for i in range(len(rects)):
        r = rects[i]
        rects[i][2] = r[0] + r[2]
        rects[i][3] = r[1] + r[3]

    # fastNonMaxSuppression
    sc = [score[0] for score in scores]
    sc = np.array(sc)


    print('rects_len', len(rects))
    print('scores:  ', scores)
    pick = fastNonMaxSuppression(rects, sc, overlapThresh=0.6)
    print('pick_len = ', len(pick))

    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

    for (x, y, xx, yy) in pick:
        print (x, y, xx, yy)
        cv2.rectangle(frame, (int(x), int(y)), (int(xx), int(yy)), (0, 0, 255), 2)

    frame = cv2.resize(frame, (640, 480))
    cv2.imshow('tracking', frame)

    k = cv2.waitKey(1) & 0xff
    if k == 27: break

cv2.destroyAllWindows()