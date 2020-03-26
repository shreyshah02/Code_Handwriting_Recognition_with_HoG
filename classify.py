from __future__ import print_function
from sklearn.externals import joblib
from hog import HOG
import dataset
import argparse
import mahotas
import cv2 as cv
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to where the model will be stored")
ap.add_argument("-i", "--image", required=True,
                help="path to the image file")
args = vars(ap.parse_args())

model = joblib.load(args["model"])

hog = HOG(orientations=18, pixelsPerCell=(10,10), cellsPerBlock=(1,1),
          transform=True)

image = cv.imread(args["image"])
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

blurred = cv.GaussianBlur(gray, (5,5), 0)
edged = cv.Canny(blurred, 30, 150)
(_, cnts, _) = cv.findContours(edged.copy(), cv.RETR_EXTERNAL,
                               cv.CHAIN_APPROX_SIMPLE)

cnts = sorted([(c, cv.boundingRect(c)[0]) for c in cnts], key =
              lambda  x: x[1])

for (c, _) in cnts:
    (x, y, w, h) = cv.boundingRect(c)

    if w>= 7 and h>=20:
        roi = gray[y:y + h, x:x + w]
        thresh = roi.copy()
        T = mahotas.thresholding.otsu(roi)
        thresh[thresh > T] = 255
        thresh = cv.bitwise_not(thresh)

        thresh = dataset.deskew(thresh, 20)
        thresh = dataset.center_extent(thresh, (20, 20))

        cv.imshow("thresh", thresh)
        hist = hog.describe(thresh)
        digit = model.predict([hist])[0]
        print("I think that number is: {}".format(digit))

        cv.rectangle(image, (x, y), (x+w, y+h),
                     (0, 255, 0), 1)
        cv.putText(image, str(digit), (x - 10, y - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv.imshow("image", image)
        cv.waitKey(0)
        #plt.imshow(image), plt.title('Image'), plt.show()

cv.imwrite('image1.jpg', image)
