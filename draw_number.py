from asyncore import loop
import numpy as npy
import cv2 as cv

def nothing(x):
 pass
 
drawing=False
def draw(event,x,y,flag,param):
 global drawing
 if event==cv.EVENT_LBUTTONDOWN:
  cv.circle(img,(x,y),8,255,-1)
  drawing=True
 elif event==cv.EVENT_MOUSEMOVE:
  if drawing==True:
   cv.circle(img, (x, y), 8, 255, -1)
 elif event==cv.EVENT_LBUTTONUP:
  cv.circle(img,(x,y),1,255,-1)
  drawing=False

def draw_number(): 
    global img
    img = npy.zeros((200,200,1),npy.uint8)
    cv.namedWindow("draw_number")
    switch="0:OFF\n1:ON"
    cv.createTrackbar(switch,"draw_number",1,1,nothing)
    while(1):
        cv.imshow("draw_number",img)
        k=cv.waitKey(1)&0xFF
        if k==27:
            break
        s = cv.getTrackbarPos(switch, "draw_number")
        if s==0:
            img = cv.resize(img,[28,28])
            img = img.reshape([1,28*28])
            cv.destroyAllWindows()
            return img
            #img[:] = 0
        else:
            cv.setMouseCallback("draw_number", draw)
        

#while(1):
#    draw_number()