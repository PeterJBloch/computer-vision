import torch
import cv2
import numpy as np

# path_to_local_pt = "./best.pt"
# model = torch.hub.load('ultralytics/yolov5', 'custom', path = path_to_local_pt)
path_to_net = "./best.onnx"
net = cv2.dnn.readNet('best.onnx')
confidence_thresh = 0.5

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
videofile = "./Aspect-adjusted_video.mp4"

def write_coords(data_string):
    with open("coords.txt", "w") as writefile:
        writefile.write(data_string + "\n")
    return

def format_yolov5(source): #Function taken from medium: https://medium.com/mlearning-ai/detecting-objects-with-yolov5-opencv-python-and-c-c7cf13d1483c
    # YOLOV5 needs a square image. Basically, expanding the image to a square
    # put the image in square big enough
    col, row, _ = source.shape
    _max = max(col, row)
    resized = np.zeros((_max, _max, 3), np.uint8)
    resized[0:col, 0:row] = source
    # resize to 640x640, normalize to [0,1[ and swap Red and Blue channels
    result = cv2.dnn.blobFromImage(resized, 1/255.0, (640, 640), swapRB=True)
    return result

def unwrap_detection(input_image, output_data):
    class_ids = []
    confidences = []
    boxes = []

    rows = output_data.shape[0]
    image_width, image_height, _ = input_image.shape

    x_factor = image_width / 360
    y_factor =  image_height / 640

    for r in range(rows):
        row = output_data[r]
        confidence = row[4]
        if confidence >= confidence_thresh:

            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if (classes_scores[class_id] > .25):

                confidences.append(confidence)
                class_ids.append(class_id)

                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

    return class_ids, confidences, boxes

#This line is changed for real-time detection so that it's the CV camera
# cap = cv2.VideoCapture(videofile)

cap = cv2.VideoCapture(0)
width, height = (640, 360)
cap.set(3, width)
cap.set(4, height)

if (cap.isOpened()== False):
    print("Error opening video stream or file")

# Originally was read until video is completed
# while(cap.isOpened()):

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret == True:
        # print("frame shape: {}".format(frame.shape))
        adjusted_image = format_yolov5(frame)
        net.setInput(adjusted_image)
        predictions = net.forward()
        output = predictions[0]
        class_ids, confidences, boxes = unwrap_detection(frame, output)
        
        #Remove duplicates using non-max suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_thresh, 0.45) 
        result_class_ids = []
        result_confidences = []
        result_boxes = []

        for i in indexes:
            result_confidences.append(confidences[i])
            result_class_ids.append(class_ids[i])
            result_boxes.append(boxes[i])

        
        # print(confidences, boxes, "\n")
        if len(confidences) > 0:
            #Order them for fast indexing later
            confidences, boxes = zip(*sorted(zip(confidences,boxes), reverse=True))

            #attempted to put in json format string
            string_to_write = "{"+"confidence: {}".format(confidences[0]) + ", box (x,y,w,h): {}".format(boxes[0])+ "}"
            write_coords(string_to_write)
            # Display the resulting frame
            box = boxes[0]
            conf = confidences[0]
            color = (255, 255, 0)
            cv2.rectangle(frame, box, color, 2)
            cv2.rectangle(frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
            cv2.putText(frame, "Apple", (box[0] + 5, box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))
        else:
            #for now do nothing
            pass
        
        cv2.imshow('Frame',frame)
	    # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

	  # Break the loop
    else:
        break
# When everything done, release the video capture object
cap.release()
# Closes all the frames
cv2.destroyAllWindows()
# predictions = net.forward()
# output = predictions[0]