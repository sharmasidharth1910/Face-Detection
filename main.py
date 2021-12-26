import cv2

#Detecting face using the haarcascade_frontalface_default ml model from the live feed using the camera

#Create a detector using the prebuilt haarcascade model
detector = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

#Capture the live camera feed, from the primary camera
cam = cv2.VideoCapture(0)

#Main Program Logic
while True:
    #ret - boolean, frame - current frame from the camera
    ret, frame = cam.read()
    
    if ret == False:
        continue
    
    #Detect all the faces from the current frame
    all_faces = detector.detectMultiScale(frame, 1.3, 5)
    
    #Draw a rectangle around all the detected faces
    for face in all_faces:
        x, y, w, h = face
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
    #Show the modified frame that contains the rectangle
    cv2.imshow("Face Detection", frame)
    
    #Termination condition for the infinite loop
    #Find the key pressed by the user
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break
    
#Release the resources
cam.release()
cv2.destroyAllWindows()