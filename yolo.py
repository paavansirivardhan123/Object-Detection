import cv2
import numpy as np

YOLO_model = cv2.dnn.readNet("yolov3.weights", "yolov3_model.cfg")
classes = []

with open("coco.names", "r") as file:
    for str in file:
        classes.append(str.strip())


yolo_output_layer = YOLO_model.getUnconnectedOutLayersNames()


img = cv2.imread("image1.jpg")

height, width, channels = img.shape

colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')




blob = cv2.dnn.blobFromImage(
    img,
    scalefactor= 1/255.0,
    size=(416,416),
    mean=(0,0,0),
    swapRB=True,
    crop=False
)
YOLO_model.setInput(blob)

outputs = YOLO_model.forward(yolo_output_layer)


classes_ids = []
boxes = []
confidences  =[] 

for output in outputs:
    for detection in output:
        scores = detection[5:]             
        class_id=np.argmax(scores) 
        confidence = scores[class_id]

        if confidence > 0.5:
            center_x = detection[0] * width
            center_y = detection[1] * height
            w = detection[2] * width
            h = detection[3] * height

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, int(w), int(h)])
            confidences.append(float(confidence))
            classes_ids.append(class_id)

Indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.7, 0.4)

for i in Indexes:
    x, y, w, h = boxes[i]
    class_id = classes_ids[i]     
    label = classes[class_id]
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x + w, y + h), color.tolist(), 1)
    cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color.tolist(), 2)


cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()