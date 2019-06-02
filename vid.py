import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
import numpy as np
import cv2
import time
from lanet import *

def get_frozen_graph(graph_file):
    """Read Frozen Graph file from disk."""
    with tf.gfile.GFile(graph_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def

def save_image(data, fname="img1.png", swap_channel=True):
    if swap_channel:
        data = data[..., ::-1]
    cv2.imwrite(fname, data)

def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=0.4, thickness=1):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]),
                  (x + size[0], y), (34, 34, 178), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale,
                (255, 255, 255), thickness)


def non_max_suppression(boxes, probs=None, nms_threshold=0.3):
    """Non-max suppression

    Arguments:
        boxes {np.array} -- a Numpy list of boxes, each one are [x1, y1, x2, y2]
    Keyword arguments
        probs {np.array} -- Probabilities associated with each box. (default: {None})
        nms_threshold {float} -- Overlapping threshold 0~1. (default: {0.3})

    Returns:
        list -- A list of selected box indexes.
    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes are integers, convert them to floats -- this
    # is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the
    # bottom-left y-coordinate)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2

    # if probabilities are provided, sort on them instead
    if probs is not None:
        idxs = probs

    # sort the indexes
    idxs = np.argsort(idxs)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value
        # to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding
        # box and the smallest (x, y) coordinates for the end of the bounding
        # box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap greater
        # than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > nms_threshold)[0])))
    # return only the bounding boxes indexes
    return pick

if __name__ == "__main__":
    
    input_names = ['image_tensor']

    # what we want from COCO labels: vehicles
    desired_classes_names = ["truck", "bus", "motorcycle", "car", "bicycle"]
    desired_classes = {3:"car", 2:"bicycle", 4:"motorcycle", 6:"bus", 8:"truck"}

    # calculation: SSD input dimension: 300 x 300; we want output window: (1280 / 720)*300 x 300
    standard_width = int((1280 / 720.0) * 300.0)
    standard_height = 300

    # The TensorRT inference graph file downloaded from Colab or your local machine.
    pb_fname = "trt_graph1.pb"
    trt_graph = get_frozen_graph(pb_fname)

    print("Loading TensorRT model, currently SSD-MobileNet-v1")

    # Create session and load graph
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_sess = tf.Session(config=tf_config)
    tf.import_graph_def(trt_graph, name='')

    tf_input = tf_sess.graph.get_tensor_by_name(input_names[0] + ':0')
    tf_scores = tf_sess.graph.get_tensor_by_name('detection_scores:0')
    tf_boxes = tf_sess.graph.get_tensor_by_name('detection_boxes:0')
    tf_classes = tf_sess.graph.get_tensor_by_name('detection_classes:0')
    tf_num_detections = tf_sess.graph.get_tensor_by_name('num_detections:0')

    tf_input.shape.as_list()


    print("TensorRT model loaded!")
    print("Loading video...")

    """
    Camera module
    """
    # print(gstreamer_pipeline(flip_method=0))
    # cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    cap = cv2.VideoCapture("project_video.mp4")
    framePerSecond = 8.0
    cv2.namedWindow('carVideo', 0)
    while(cap.isOpened()):
        _, image = cap.read()
        now = time.time()

        """
        Inside the main processing loop, do stuff
        """

        image_to_show = cv2.resize(image, (standard_width, standard_height))
        try:
            image_to_show = process_image(image_to_show)
        except:
            print("error detecting lanes...")
        image = cv2.resize(image, (300, 300))
        
        print("image processing finished, inferencing...")
        """
        Original inference step
        """
        scores, boxes, classes, num_detections = tf_sess.run([tf_scores, tf_boxes, tf_classes, tf_num_detections], feed_dict={
            tf_input: image[None, ...]
        })
        boxes = boxes[0]  # index by 0 to remove batch dimension
        scores = scores[0]
        classes = classes[0]
        # num_detections = int(num_detections[0])


        # retain only those detection we are interested in: vehicles
        desired_detection_idx = np.isin(classes, list(desired_classes.keys()))
        boxes = boxes[desired_detection_idx]
        scores = scores[desired_detection_idx]
        classes = classes[desired_detection_idx]
        num_detections = len(scores)

        # Boxes unit in pixels (image coordinates).
        boxes_pixels = []
        for i in range(num_detections):
            # scale box to image coordinates
            box = boxes[i] * np.array([image_to_show.shape[0],
                                    image_to_show.shape[1], image_to_show.shape[0], image_to_show.shape[1]])
            box = np.round(box).astype(int)
            boxes_pixels.append(box)
        boxes_pixels = np.array(boxes_pixels)

        # Remove overlapping boxes with non-max suppression, return picked indexes.
        pick = non_max_suppression(boxes_pixels, scores[:num_detections], 0.5)
        # print(pick)

        print("non max suppression finished.")

        for i in pick:
            box = boxes_pixels[i]
            box = np.round(box).astype(int)
            # Draw bounding box.
            image_to_show = cv2.rectangle(
                image_to_show, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 2)
            # label = "{}:{:.2f}".format(int(classes[i]), scores[i])
            label = desired_classes[classes[i]] + " {:.2f}".format(scores[i])
            # Draw label (class index and probability).
            draw_label(image_to_show, (box[1], box[0] - 2), label)

        """
        End of original inference step
        """

        cv2.imshow('video input result',image_to_show)
        # This also acts as 
        keyCode = cv2.waitKey(10) & 0xff
        # Stop the program on the ESC key
        if keyCode == 27:
            break


        """
        End of processing, check elapsed time
        """

        elapsed = time.time() - now
        if elapsed < 1 / framePerSecond:
            time.sleep((1 / framePerSecond) - elapsed)

    tf_sess.close()
    cap.release()
    cv2.destroyAllWindows()

    pass
