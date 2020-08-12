import dlib
import cv2
import numpy as np
import os
from blinker import process_face, load_model, isBlinking
from mobile_net import load_ssd_model, boxes_from_image, custom_iou

# Create the tracker we will use
tracker = dlib.correlation_tracker()

# The variable we use to keep track of the fact whether we are
# currently using the dlib tracker
trackingFace = 0
blink_string = ''
blinked = False

video_capture = cv2.VideoCapture(0)

protoPath = os.path.sep.join(['face_detector', "deploy.prototxt"])
modelPath = os.path.sep.join(['face_detector', "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

open_eye_cascPath = os.path.sep.join(['cascade', 'haarcascade_eye_tree_eyeglasses.xml'])
left_eye_cascPath = os.path.sep.join(['cascade', 'haarcascade_lefteye_2splits.xml'])
right_eye_cascPath = os.path.sep.join(['cascade', 'haarcascade_righteye_2splits.xml'])

open_eyes_detector = cv2.CascadeClassifier(open_eye_cascPath)
left_eye_detector = cv2.CascadeClassifier(left_eye_cascPath)
right_eye_detector = cv2.CascadeClassifier(right_eye_cascPath)

eye_model = load_model()

ssd_model = load_ssd_model()

record = False
if record:
    output='o.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output, fourcc, 20.0, (width, height))

while True:
    ret, frame = video_capture.read()

    if not ret:
        break
    img = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    # fontScale
    fontScale = 1
    cv2.putText(img, str(trackingFace), (20, 20), font, fontScale, (0, 255, 0), thickness=2)
    if trackingFace == 2:
        trackingQuality = tracker.update(img)

        if trackingQuality >= 8.75:
            tracked_position = tracker.get_position()

            t_x = int(tracked_position.left())
            t_y = int(tracked_position.top())
            t_w = int(tracked_position.width())
            t_h = int(tracked_position.height())

            cv2.rectangle(img, (t_x, t_y),
                          (t_x + t_w, t_y + t_h),
                          (255, 0, 0), 2)
            cv2.putText(img, 'real' if blinked else 'fake', (t_x - 5, t_y - 5), font, fontScale, (255, 0, 0),
                        thickness=2)

        else:
            trackingFace = 0
    if trackingFace == 1:
        trackingQuality = tracker.update(img)

        if trackingQuality >= 8.75:
            tracked_position = tracker.get_position()

            t_x = int(tracked_position.left())
            t_y = int(tracked_position.top())
            t_w = int(tracked_position.width())
            t_h = int(tracked_position.height())
            if t_x + t_w > frame.shape[1] or t_y + t_h > frame.shape[0]:
                trackingFace = 0
            else:
                if not blinked:
                    blink_string = process_face(blink_string, frame, t_x, t_y, t_w, t_h,
                                                open_eyes_detector,
                                                left_eye_detector, right_eye_detector, eye_model)
                    blinked = isBlinking(blink_string, 0, 3)
                    if len(blink_string) > 50:
                        print(blink_string[-50:])
                    else:
                        print(blink_string)
                    if blinked:
                        boxes = boxes_from_image(ssd_model, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

                        for obj_box in boxes:
                            print(obj_box, (t_x, t_y, t_w, t_h))
                            iou = custom_iou(obj_box, (t_x, t_y, t_w, t_h))
                            print(iou)
                            if iou > 0.1:
                                print('in cell phone')
                                blinked = False
                                trackingFace = 2
                                break
                cv2.rectangle(img, (t_x, t_y),
                              (t_x + t_w, t_y + t_h),
                              (0, 0, 255) if blinked else (255, 0, 0), 2)
                cv2.putText(img, 'real' if blinked else 'fake', (t_x - 5, t_y - 5), font, fontScale, (255, 0, 0),
                            thickness=2)

        else:
            trackingFace = 0
    if trackingFace == 0:

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        # ensure at least one face was found
        if len(detections) > 0:

            i = np.argmax(detections[0, 0, :, 2])

            confidence = detections[0, 0, i, 2]
            if confidence > 0.8:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

                (startX, startY, endX, endY) = box.astype("int")

                face = frame[startY:endY, startX:endX]
                skip = False
                if endY > startY and endX > startX:
                    boxes = boxes_from_image(ssd_model, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

                    for obj_box in boxes:
                        iou = custom_iou(obj_box, box.astype("int"))
                        print(iou)
                        if iou > 0.1:
                            print('in cell phone')
                            skip = True
                            break

                    cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0))
                    # Initialize the tracker
                    tracker.start_track(img,
                                        dlib.rectangle(startX - 10,
                                                       startY - 20,
                                                       endX + 10,
                                                       endY + 20))

                    if not skip:
                        trackingFace = 1
                        blink_string = ''
                        blink_string = process_face(blink_string, frame, startX, startY, endX - startX, endY - startY,
                                                    open_eyes_detector,
                                                    left_eye_detector, right_eye_detector, eye_model)
                        blinked = False
                    else:
                        trackingFace = 2
                        blinked = False
    cv2.imshow('face', img)
    # write the flipped frame
    if record:
        out.write(img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
if record:
    out.release()
cv2.destroyAllWindows()
