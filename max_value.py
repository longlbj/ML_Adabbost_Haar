import numpy as np
import cv2
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from curvefit import curvefit 
from curvefit import curvefit_3

hyper_cascade = cv2.CascadeClassifier('/home/longzhijun/Documents/AI/GPR+AI/haar/output/cascade_3.xml')
#3.png is the best
#img = cv2.imread('/home/longzhijun/Documents/AI/GPR+AI/haar/rebar/3.png',0)
img = cv2.imread('/home/longzhijun/Documents/AI/GPR+AI/haar/rebar/3.png',0)

faces = hyper_cascade.detectMultiScale(img, 2, 5)
clone = img.copy()
x_p = []
y_p = []

for (x, y, w, h) in faces:
    sub = clone[y-1*h/5:y+4*h/5, x-2*w/5:x+3*w/5]
    a = sub.argmax()
    (m,n) = sub.shape
    x_p = x-2*w/5+a%n;
    y_p = y-1*h/5+a/n
    cv2.line(img, (x_p, y_p), (x_p, y_p), (255,255,255), 9)
    color = np.array((255,250,255))
    cv2.rectangle(img, (x-2*w/5, y-1*h/5), (x + 3*w/5, y + 4*h/5), color, 2)
cv2.imshow('img', img)
cv2.imwrite('./max.jpg',img)
cv2.waitKey(0)
#cv2.destroyAllWindows()
