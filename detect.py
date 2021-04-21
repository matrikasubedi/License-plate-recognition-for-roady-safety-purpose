import cv2
import numpy as np

net = cv2.dnn.readNet('custom.weights','yolov3_custom.cfg')
classes = []
with open('obj.names','r') as f:
    classes = f.read().splitlines()

img = cv2.imread('car.jpg')
height,width,_ = img.shape

blob = cv2.dnn.blobFromImage(img,0.003921,(416,416),(0,0,0),swapRB=True,crop=False)
net.setInput(blob)
output_layers_names = net.getUnconnectedOutLayersNames()
layersOutputs = net.forward(output_layers_names)

boxes = []
confidences = []
class_ids = []

print(len(layersOutputs))


for output in layersOutputs:
    print(output.shape)
    for detection in output:
        #print(detection)
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        #print(confidence)
        if confidence > 0.5:
            center_x = int(detection[0]*width)
            center_y = int(detection[1]*height)
            w = int(detection[2]*width)
            h = int(detection[3]*height)

            x = int(center_x - w/2)
            y = int(center_y - h/2)

            boxes.append([x,y,w,h])
            confidences.append((float(confidence)))
            class_ids.append(class_id)


indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)
print(indexes)

font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0,255,size=(len(boxes),3))

for i in indexes.flatten():
    x,y,w,h = boxes[i]
    label = str(classes[class_ids[i]])
    confidence = str(round(confidences[i],2))
    color =colors[i]
    cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
    cv2.putText(img, label + " " + confidence, (x, y-5), font, 2, (255,255,0), 2)




cv2.imshow('Image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()