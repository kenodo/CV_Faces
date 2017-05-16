import face_recongizer
import cv2

faces = "D:/testpics/output"
compareImagePath = "D:/forcompare.jpg"
original = cv2.imread(compareImagePath)
cv2.imshow("Сравнение: ", original)
face_recongizer.compareFaces(compareImagePath,faces)