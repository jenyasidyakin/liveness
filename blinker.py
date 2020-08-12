from tensorflow.keras.models import model_from_json
import numpy as np
from PIL import Image
import cv2

IMG_SIZE = 24


def load_model():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return loaded_model


def predict(img, model):
    img = np.array(Image.fromarray(img, 'RGB').convert('L').resize((IMG_SIZE, IMG_SIZE)), dtype='float32')
    # img = imresize(img, (IMG_SIZE, IMG_SIZE)).astype('float32')
    img /= 255
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    prediction = model.predict(img)
    if prediction < 0.1:
        prediction = 'closed'
    elif prediction > 0.9:
        prediction = 'open'
    else:
        prediction = 'idk'
    return prediction


def process_face(blick_string, frame, x, y, w, h, open_eyes_detector, left_eye_detector, right_eye_detector, model):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_face = gray[y:y + h, x:x + w]

    open_eyes_glasses = open_eyes_detector.detectMultiScale(
        gray_face,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # if open_eyes_glasses detect eyes then they are open
    if len(open_eyes_glasses) == 2:
        blick_string += '1'
        # for (ex, ey, ew, eh) in open_eyes_glasses:
        #     cv2.rectangle(face, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # otherwise try detecting eyes using left and right_eye_detector
    # which can detect open and closed eyes
    else:
        # separate the face into left and right sides
        left_face = frame[y:y + h, x + int(w / 2):x + w]
        left_face_gray = gray[y:y + h, x + int(w / 2):x + w]

        right_face = frame[y:y + h, x:x + int(w / 2)]
        right_face_gray = gray[y:y + h, x:x + int(w / 2)]

        # Detect the left eye
        left_eye = left_eye_detector.detectMultiScale(
            left_face_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Detect the right eye
        right_eye = right_eye_detector.detectMultiScale(
            right_face_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        eye_status = '1'  # we suppose the eyes are open

        # For each eye check wether the eye is closed.
        # If one is closed we conclude the eyes are closed
        for (ex, ey, ew, eh) in right_eye:
            color = (0, 255, 0)
            pred = predict(right_face[ey:ey + eh, ex:ex + ew], model)
            if pred == 'closed':
                eye_status = '0'
        for (ex, ey, ew, eh) in left_eye:
            color = (0, 255, 0)
            pred = predict(left_face[ey:ey + eh, ex:ex + ew], model)
            if pred == 'closed':
                eye_status = '0'
        blick_string += eye_status

    return blick_string


def isBlinking(history, minFrames, maxFrames):
    """ @history: A string containing the history of eyes status
         where a '1' means that the eyes were closed and '0' open.
        @maxFrames: The maximal number of successive frames where an eye is closed """
    for i in range(minFrames, maxFrames):
        pattern = '1' + '0' * (i + 1) + '1'
        if pattern in history:
            return True
    return False


if __name__ == '__main__':
    import os

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

    model = load_model()

    blink_string = ''
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        img = frame.copy()
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

                blink_string = process_face(blink_string, frame, startX, startY, endX - startX, endY - startY,
                                            open_eyes_detector,
                                            left_eye_detector, right_eye_detector, model)

                print(blink_string)

        cv2.imshow('face', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
