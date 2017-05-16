#!/usr/bin/python

# Import the required modules
import cv2, os
import numpy as np
from PIL import Image
import main

# For face detection we will use the Haar Cascade provided by OpenCV.
cascadePath = "D:/testpics/haar.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# For face recognition we will the the LBPH Face Recognizer
recognizer = cv2.face.createLBPHFaceRecognizer()






def get_images_and_labels(path):
    # Append all the absolute image paths in a list image_paths
    # We will not read the image with the .sad extension in the training set
    # Rather, we will use them to test our accuracy of the training
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if not f.endswith('.sad')]
    # images will contains face images
    images = []
    # labels will contains the label that is assigned to the image
    labels = []
    for image_path in image_paths:
        # Read the image and convert to grayscale
        image_pil = cv2.imread(image_path)
        image_pil = cv2.cvtColor(image_pil, cv2.COLOR_BGR2GRAY)
        # Convert the image format into numpy array
        image = np.array(image_pil, 'uint8')
        # Get the label of the image
        nbr = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
        # Detect the face in the image
        faces = faceCascade.detectMultiScale(image)
        # If face is detected, append the face to images and the label to labels
        for (x, y, w, h) in faces:
            images.append(image[y: y + h, x: x + w])
            labels.append(nbr)
            cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
            cv2.waitKey(5)
    # return the images list and labels list
    return images, labels


def compareFaces(imagePath, facesPath):
    # Path to the Yale Dataset

    images, labels = get_images_and_labels(facesPath)
    cv2.destroyAllWindows()

    # Perform the tranining
    recognizer.train(images, np.array(labels))

    # Append the images with the extension .sad into image_paths
    image_path = imagePath

    predict_image_pil = cv2.imread(image_path)
    predict_image_pil = cv2.resize(predict_image_pil, (300,300))
    predict_image_pil = cv2.cvtColor(predict_image_pil, cv2.COLOR_BGR2GRAY)
    predict_image = np.array(predict_image_pil, 'uint8')
    faces = faceCascade.detectMultiScale(predict_image)
    for (x, y, w, h) in faces:
        result = cv2.face.MinDistancePredictCollector()
        recognizer.predict(predict_image, result, 0)
        nbr_predicted = result.getLabel()
        conf = result.getDist()

        nbr_actual = image_path
        print (str(nbr_actual) + " is Recognized as " + str(nbr_predicted) + " with conf " + str(conf))


        predictedFace = cv2.imread(facesPath + '/' + str(nbr_predicted) + '.jpg')
        cv2.imshow('Predicted: ', predictedFace)
        cv2.imshow("Recognizing Face", predict_image)
        cv2.waitKey(1000)
