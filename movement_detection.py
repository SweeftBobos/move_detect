import cv2 
import numpy as np 
cap = cv2.VideoCapture(0) 
 
correct, old_frame = cap.read() 
correct, new_frame = cap.read() 
 
while cap.isOpened: 
 
    odd = cv2.absdiff(old_frame, new_frame) 
    gray_frame = cv2.cvtColor(odd, cv2.COLOR_BGR2GRAY)  
    gray_gauss_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0) 
    _, thres = cv2.threshold(gray_gauss_frame, 20, 255, cv2.THRESH_BINARY) 
    kernel = np.ones((10, 15), 'uint8') 
    dil = cv2.dilate(thres, kernel) 
    contours, _ = cv2.findContours(dil, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
 
    final_contour = cv2.drawContours(old_frame, contours, -1, (0,0,255), -1, cv2.LINE_AA) 
     
 
 
    cv2.imshow("old_frame", old_frame) 
    old_frame = new_frame 
 
    correct, new_frame = cap.read() 
 
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break  
 
 
cap.release() 
cv2.destroyAllWindows()