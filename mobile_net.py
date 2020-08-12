import tensorflow as tf
import numpy as np


def load_ssd_model():
    model_dir = "saved_model"
    model = tf.saved_model.load(model_dir)
    model = model.signatures['serving_default']

    return model


def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    output_dict = model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    return output_dict


def boxes_from_image(model, image_np):
    res = []
    output_dict = run_inference_for_single_image(model, image_np)
    for i, ii in enumerate(output_dict['detection_classes']):
        cla = output_dict['detection_classes'][i]
        # cl = df[df.id == cla]['label'].values[0]
        if output_dict['detection_scores'][i] > 0.1:

            if cla == 73 or cla == 77:
                y1 = int(output_dict['detection_boxes'][i][0] * image_np.shape[0])
                x1 = int(output_dict['detection_boxes'][i][1] * image_np.shape[1])

                y2 = int(output_dict['detection_boxes'][i][2] * image_np.shape[0])
                x2 = int(output_dict['detection_boxes'][i][3] * image_np.shape[1])

                res.append((x1, y1, x2, y2))

    return res


def custom_iou(box1, face_box):  # iou between face and cellphone
    # push up all the lower origin-xs, while keeping the higher origin-xs
    ys1 = np.maximum(box1[0], face_box[0])
    # get all the origin-xs
    # push right all the lower origin-xs, while keeping higher origin-xs
    xs1 = np.maximum(box1[1], face_box[1])
    # get all the target-ys
    # pull down all the higher target-ys, while keeping lower origin-ys
    ys2 = np.minimum(box1[2], face_box[2])
    # get all the target-xs
    # pull left all the higher target-xs, while keeping lower target-xs
    xs2 = np.minimum(box1[3], face_box[3])

    box_area2 = (face_box[3] - face_box[1]) * (face_box[2] - face_box[0])
    intersections = np.maximum(ys2 - ys1, 0) * np.maximum(xs2 - xs1, 0)
    # each union is then the box area
    # added to each other box area minusing their intersection calculated above
    return 0 if box_area2 == 0 else intersections / box_area2


if __name__ == '__main__':

    import cv2
    import pandas as pd

    df = pd.read_csv('label_df.csv')
    video_capture = cv2.VideoCapture(0)

    model = load_ssd_model()

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        img = frame.copy()
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_np = np.array(img2)

        output_dict = run_inference_for_single_image(model, image_np)
        for i, ii in enumerate(output_dict['detection_classes']):
            cla = output_dict['detection_classes'][i]
            cl = df[df.id == cla]['label'].values[0]
            if output_dict['detection_scores'][i] > 0.1:
                if cl == 'cell phone' or cl == 'laptop':
                    y1 = int(output_dict['detection_boxes'][i][0] * image_np.shape[0])
                    x1 = int(output_dict['detection_boxes'][i][1] * image_np.shape[1])

                    y2 = int(output_dict['detection_boxes'][i][2] * image_np.shape[0])
                    x2 = int(output_dict['detection_boxes'][i][3] * image_np.shape[1])

                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0))

                    font = cv2.FONT_HERSHEY_SIMPLEX

                    # fontScale
                    fontScale = 1
                    cv2.putText(img, cl, (x1, y1), font, fontScale, (0, 255, 0), thickness=2)
        cv2.imshow('face', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
