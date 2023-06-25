import cv2


faceProto="opencv_face_detector.pbtxt"
faceModel="opencv_face_detector_uint8.pb"
genderProto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"
padding=20
conf_threshold=0.7
meanvalues=(78.4263377603, 87.7689143744, 114.895847746)

genders=['Male','Female']

faceDetect=cv2.dnn.readNet(faceModel,faceProto)
genderDetect=cv2.dnn.readNet(genderModel,genderProto)

img_input=input("image path ? : ")

try:
    image=cv2.VideoCapture(img_input)
    hasFrame,frame=image.read()
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    faceDetect.setInput(blob)
    detections=faceDetect.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    if not faceBoxes:
       print("No faces were detected")
    for x in faceBoxes:
        detected_face=frame[max(0,x[1]-padding):
                  min(x[3]+padding,frame.shape[0]-1),max(0,x[0]-padding)
                  :min(x[2]+padding, frame.shape[1]-1)]
        blob=cv2.dnn.blobFromImage(detected_face, 1.0, (227,227), meanvalues, swapRB=False)
        genderDetect.setInput(blob)
        genderPrediction=genderDetect.forward()[0].argmax()
        genderpredicted=genders[genderPrediction]
        print("Predicted Gender is "+genderpredicted)
except Exception as e:
    print(e)
    pass
