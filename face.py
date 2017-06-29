import sys
import cv2

imagePath = [sys.argv[1], sys.argv[2]]
cascFacePath = '/home/burakcl/PycharmProjects/yuztanima/haarcascade_frontalface_default.xml'
cascEyePath = '/home/burakcl/PycharmProjects/yuztanima/haarcascade_eye.xml'
cascMouthPath = '/home/burakcl/PycharmProjects/yuztanima/haarcascade_smile.xml'
face_Cascade = cv2.CascadeClassifier(cascFacePath)
eye_Cascade = cv2.CascadeClassifier(cascEyePath)
mou_cascade = cv2.CascadeClassifier(cascMouthPath)

def detect_face(path):
    yuzagiz = []
    yuzgoz = []
    im = cv2.imread(path)
    img = cv2.resize(im, (500, 500))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    faces = face_Cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        img_gray = gray[y:y + h, x:x + w]
        img_color = img[y:y + h, x:x + w]
        eyes = eye_Cascade.detectMultiScale(
            img_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(img_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            mouth = mou_cascade.detectMultiScale(
                img_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.cv.CV_HAAR_SCALE_IMAGE
            )
            for (mx, my, mw, mh) in mouth:
                cv2.rectangle(img_color, (mx, my), (mx + mw, my + mh), (0, 0, 255), 2)
                yuzagiz.append(float(mouth[0][3]) / float(faces[0][3]))
                yuzgoz.append(float(eyes[0][3]) / float(faces[0][3]))

    return img, yuzagiz, yuzgoz

def es(yao, ygo):

    if 0.80 <= yao < 1.2 and 0.80 <= ygo < 1.2:
        print "eslesti"
    else:
        print "eslesmedi"

    return yao, ygo

yuz = detect_face(sys.argv[1])
yuz1 = detect_face(sys.argv[2])
if yuz[1] and yuz1[1]:
    yao = float(yuz1[1][1])/float(yuz[1][1])
    ygo = float(yuz1[2][1])/float(yuz[2][1])
    es(yao, ygo)
    final = cv2.hconcat((yuz[0], yuz1[0]))
    cv2.imshow("gv",final)
    cv2.waitKey(0)
else:
    print "yuz tespit edilemedi"
