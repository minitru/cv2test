import cv2
import numpy as np

# CAN USE MUCH DEEPER MODEL (10x)
net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net) 
model.setInputParams(size=(320,320),scale=1/255)

# load class list
classes = []
with open("dnn_model/classes.txt", "r") as file_object:
	for class_name in file_object.readlines():
		class_name = class_name.strip()
		classes.append(class_name)

# print("Objects List")
# print(classes)

#init camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# FULL HD 1920 x 1080

button_person = False


def click_button(event, x, y, flags, params):
	global button_person
	if event == cv2.EVENT_LBUTTONDOWN: 
		print(x,y)
		polygon = np.array([[(20,20), (220,20), (220, 70), (20, 70)]])

		is_inside = cv2.pointPolygonTest(polygon, (x,y), False)
		if is_inside > 0:
			print ("INSIDE THE BUTTON")
			if button_person is False:
				button_person = True
			else:
				button_person = False
		print("Now button person is: ", button_person)


# Create window
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", click_button)

while True:
	ret, frame = cap.read()

	# object detection
	# class 0 = person, class 2 = car
	
	# bbox is top left + length, width
	(class_ids, scores, bboxes) = model.detect(frame)
	for class_id, score, bbox in zip(class_ids, scores, bboxes):
		(x, y, w, h) = bbox

		class_name = classes[class_id]
		if class_name == "person" and button_person is True: 
			# print(x, y, w, h)
			cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (200,0,50), 2)
			cv2.rectangle(frame, (x, y), (x+w, y+h), (200, 0, 50), 3)

	# Create button
	cv2.rectangle(frame, (20,20), (220, 70), (0, 0, 200), -1)
	polygon = np.array([[(20,20), (220,20), (220, 70), (20, 70)]])
	cv2.fillPoly(frame, polygon, (0, 0, 200))
	cv2.putText(frame, "person", (30,60), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 3)
	
	# print("class ids", class_ids)
	# print("scores", scores)
	# print("bboxes", bboxes)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(100)
	print("KEY: ", key)

	# EXIT ON 'q'
	if key == 113:
		break

cap.release()
cv2.destroyAllWindows()
