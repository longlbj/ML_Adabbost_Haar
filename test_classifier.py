import numpy as np
import cv2

hyper_cascade = cv2.CascadeClassifier('/home/longzhijun/Documents/AI/GPR+AI/haar/output/cascade_3.xml')
#3.png is the best
#img = cv2.imread('/home/longzhijun/Documents/AI/GPR+AI/haar/rebar/3.png',0)
img = cv2.imread('/home/longzhijun/Documents/AI/GPR+AI/haar/rebar/3.png',0)

faces = hyper_cascade.detectMultiScale(img, 2, 5)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x-2*w/5, y-1*h/5), (x + 3*w/5, y + 4*h/5), (255, 0, 0), 2)
    #cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

cv2.imshow('img', img)
cv2.imwrite('./new.jpg', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
