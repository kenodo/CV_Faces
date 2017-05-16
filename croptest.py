import cv2
import os, os.path
import main

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades
# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml

face_cascade = cv2.CascadeClassifier('D:/testpics/haar.xml')



forCompare = cv2.imread('D:/forcompare.jpg')
cv2.imshow('Comparing..', forCompare)
forCompare = cv2.resize(forCompare, (300, 300))




DIR = 'D:/testpics'
maxPercent = 10000
name=''
numPics = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
print(numPics)

for pic in range(1, (numPics)):
    img = cv2.imread('D:/testpics/'+str(pic)+'.jpg')
    height = img.shape[0]
    width = img.shape[1]
    size = height * width


    if size > (500^2):
        r = 500.0 / img.shape[1]
        dim = (500, int(img.shape[0] * r))
        img2 = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        img = img2

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # подключаем каскады Хаара

    for (x,y,w,h) in faces:
        imgCrop = img[y:y+h,x:x+w]
        #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        imgCrop2 = cv2.resize(imgCrop, (300, 300))
        cv2.imwrite("D:/testpics/output/"+str(pic)+".jpg", imgCrop2)

        #
        #
        # Ниже - сверяем изображения
        #

        similarity = main.compare_images_silent(imgCrop2,forCompare)
        #
        #
        #
        #

        if (similarity<maxPercent):
            maxPercent=similarity
            name=str(pic) + '.jpg'

    cv2.imshow('img',imgCrop)
    print("Image"+str(pic)+" has been processed and cropped")
    cv2.waitKey(1)

cv2.destroyWindow('img')                    # Удаляем последнее окно с сравниваемой картинкой

print('')
print('//////////////////////////////////////////////')
print("Все изображения были обработаны.")
print('Самая похожая: ' + name)
theMost = cv2.imread('D:/testpics/' + name)
cv2.imshow("The most similar: ", theMost)
print('Ошибка: ' + str(maxPercent))         # Процент ошибки
print('//////////////////////////////////////////////')
cv2.waitKey(5000)
#cv2.destroyAllWindows()
#cv2.destroyAllWindows()