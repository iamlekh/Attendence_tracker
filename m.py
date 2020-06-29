import cv2
import numpy as np
import face_recognition

def load_rezize(img):
    image = cv2.imread(img)
    stretch_near = cv2.resize(image, (500,500),interpolation = cv2.INTER_NEAREST)
    cv2.imwrite(img , stretch_near)
    return img


img_darpan = face_recognition.load_image_file(load_rezize("mridu.jpg"))
img_darpan = cv2.cvtColor(img_darpan,cv2.COLOR_BGR2RGB)
img_test = face_recognition.load_image_file(load_rezize("darpan.jpg"))
img_test = cv2.cvtColor(img_test,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(img_darpan)[0]
encodeddarpan = face_recognition.face_encodings(img_darpan)[0]
cv2.rectangle(img_darpan,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(225,225,0), 2)
print(faceLoc)

faceLoc = face_recognition.face_locations(img_test)[0]
encodedtest = face_recognition.face_encodings(img_test)[0]
cv2.rectangle(img_test,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(225,100,0), 2)
print(faceLoc)

result  = face_recognition.compare_faces([encodeddarpan], encodedtest)
faceDist  = face_recognition.face_distance([encodeddarpan], encodedtest)
cv2.putText(img_test,f'{result} {round(faceDist[0],2)} ',(30,30),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,205),2)
print(result, faceDist)



cv2.imshow('sAMPLE',img_darpan)
cv2.imshow('test',img_test)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("end")

