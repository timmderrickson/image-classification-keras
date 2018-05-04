# USAGE
# python test_network.py --model santa_not_santa.model --image images/examples/santa_01.png

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to person or not trained model")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-h", "--happy", required=True,
	help='path to happy or not trained model')
args = vars(ap.parse_args())

# load the image
image = cv2.imread(args["image"])
orig = image.copy()

# pre-process the image for classification
image = cv2.resize(image, (28, 28))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model(args["model"])
modelh = load_model(args['happy'])

# classify the input image
(notPerson, person) = model.predict(image)[0]
(notHappy, happy) = modelh.predict(image)[0]

# build the labels
label_person = "Person" if person > notPerson else "Not Person"
proba = person if person > notPerson else notPerson
label_person = "{}: {:.2f}%".format(label_person, proba * 100)

label_happy = "Happy" if happy > notHappy else "Not Happy"
proba_happy = happy if happy > notHappy else notHappy
if label_happy == happy:
	label_happy = "{}: {:.2f}%".format(label_happy, proba_happy * 100)
else:
	label_happy = "Things don't feel"

# draw the label on the image
output = imutils.resize(orig, width=400)
cv2.putText(output, label_person, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 255, 0), 2)

cv2.putText(output, label_happy, (0, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 255, 0), 2)

# show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)