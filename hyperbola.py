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
    m,n = sub.shape
    for i in range(m):
        for j in range(n):
            if (sub[i,j]<150):
                sub[i,j] = 0
            else:
                sub[i,j] = 1
                y_p.append(i)
                x_p.append(j)
    #plt.imshow(sub,plt.cm.gray)
    #plt.scatter(x_p, y_p, marker='x', color='m')
    #plt.colorbar()
    #plt.show()
    #A1, B1, C1 = curve_fit(curvefit, x_p, y_p)[0]
    A1, B1, C1, D1 = curve_fit(curvefit_3, x-2*w/5+x_p, y-1*h/5+y_p)[0]
    print A1, B1, C1, D1
    x_c = np.arange(x-2*w/5, x+3*w/5,0.1)
    y_c = A1*x_c*x_c*x_c + B1*x_c*x_c + C1*x_c + D1
    pts = np.int32(np.column_stack((x_c, y_c)))
    img = cv2.polylines(img,[pts],False,(0, 0, 0))
    color = np.array((255,250,255))
    cv2.rectangle(img, (x-2*w/5, y-1*h/5), (x + 3*w/5, y + 4*h/5), color, 2)
cv2.imshow('img', img)
cv2.imwrite('./new.jpg',img)
cv2.waitKey(0)
#cv2.destroyAllWindows()
