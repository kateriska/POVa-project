import time
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import tensorflow as tf
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import cv2
import argparse

matplotlib.use('TkAgg')

PATH_TO_RCNN_MODEL_DIR="trained_model/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8"

PATH_TO_SSD_MODEL_DIR= "" #TODO 

PATH_TO_LABELS = "annotations/label_map.pbtxt"

def parse_args():
    argParser = argparse.ArgumentParser(description='Detect traffic signs and predict probability of their class on a set of images or a video input')
    subparsers = argParser.add_subparsers(dest='subcommand', required=True)

    imgParser = subparsers.add_parser('image')
    imgParser.add_argument('-i', '--input', action='store', default='data/predict', help='Input folder with images')
    imgParser.add_argument('-o', '--output', action='store', default='output_predict/', help='Output folder for images')
    
    imgParser = subparsers.add_parser('video')
    imgParser.add_argument('-i', '--input', action='store', default='test_video.mp4', help='Input video file name')
    imgParser.add_argument('-o', '--output', action='store', default='out_iou_video.mp4', help='Output video file name')

    argParser.add_argument('-m', '--model', action='store', default="faster_rcnn", choices=["faster_rcnn", "ssd"], help='Select type of model')
    return argParser.parse_args()

def main():
    args = parse_args()

    detection_model = load_model(args.model)

    if args.subcommand == 'image':
        image_predict(detection_model, args.input, args.output)
    if args.subcommand == 'video':
        video_predict(detection_model, args.input, args.output)

def load_model(model):

    if model == 'faster_rcnn':
        PATH_TO_MODEL_DIR = PATH_TO_RCNN_MODEL_DIR
    if model == 'ssd':
        PATH_TO_MODEL_DIR = PATH_TO_SSD_MODEL_DIR
    
    PATH_TO_CFG = PATH_TO_MODEL_DIR + "/pipeline.config"
    PATH_TO_CKPT = PATH_TO_MODEL_DIR #+ "/checkpoint"
    CHECKPOINT = "ckpt-26"

    print('Loading model... ', end='')
    start_time = time.time()

    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
    model_config = configs['model']
    detection_model = model_builder.build(model_config=model_config, is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(PATH_TO_CKPT, CHECKPOINT)).expect_partial()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Model loaded in {} seconds'.format(elapsed_time))

    return detection_model

def image_predict(detection_model, input_dir, output_dir):
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                        use_display_name=True)

    IMAGE_PATHS = absoluteFilePaths(input_dir)

    for image_path in IMAGE_PATHS:

        print('Running inference for {}... '.format(image_path), end='')

        image_np = load_image_into_numpy_array(image_path)
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)

        detections = detect_fn(input_tensor, detection_model)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=20,
                min_score_thresh=.4,
                agnostic_mode=False)

        plt.figure()
        plt.axis('off')
        plt.imshow(image_np_with_detections)

        pth,fn = os.path.split(image_path)
        fn = fn.replace(".jpg",".png")
        plt.savefig("{}{}".format(output_dir,fn),transparent=True,dpi = 500, bbox_inches='tight', pad_inches = 0)
        print('Done')

def video_predict(detection_model, input_file, output_file):
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                        use_display_name=True)

    videoCapture = cv2.VideoCapture(input_file)
    videoWriter = cv2.VideoWriter(output_file,cv2.VideoWriter_fourcc(*'XVID'),12,(1392,512))

    if not videoCapture.isOpened():
        print("Error: Unable to open video file for reading", input_file)
        exit(-1)

    prev_detection = ()
    while videoCapture.isOpened():
        ret, frame = videoCapture.read()
        if not ret:
            break

        image_np = np.array(frame)
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)

        detections = detect_fn(input_tensor, detection_model)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))

        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        detections = filter_detection(prev_detection, detections,score_t=0.3,iou_t=0.1)

        if num_detections > 0:

            label_id_offset = 1
            image_np_with_detections = image_np.copy()

            viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes'] + label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=20,
                min_score_thresh=.5,
                agnostic_mode=False)

            cv2.imshow('out',image_np_with_detections)
            videoWriter.write(image_np_with_detections)
        else:
            cv2.imshow('out',image_np)
            videoWriter.write(image_np)

        prev_detection = (detections['detection_boxes'],detections['detection_classes'],detections['detection_scores'])

        if cv2.waitKey(10) == 27:
            break

    cv2.destroyAllWindows()
    videoCapture.release()
    videoWriter.release()

def intersection_over_union(box1,box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = (x2 - x1) * (y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return inter / float(box1_area + box2_area - inter)

def filter_detection(prev, det,score_t=0.7,iou_t=0.8):
    if not len(prev):
        return det
    prev_boxes, prev_classes, prev_scores = prev


    boxes = det['detection_boxes']
    classes = det['detection_classes']
    scores = det['detection_scores']

    idx = np.where(scores > score_t)
    scores = scores[idx]
    classes = classes[idx]
    boxes = boxes[idx]


    for a in range(boxes.shape[0]):
        for b in range(prev_boxes.shape[0]):
            iou = intersection_over_union(boxes[a],prev_boxes[b])
            if iou > iou_t:
                if scores[a] < prev_scores[b]:
                    scores[a] = prev_scores[b]
                    classes[a] = prev_classes[b]

    det['detection_boxes'] = boxes
    det['detection_classes'] = classes
    det['detection_scores'] = scores

    return det

@tf.function
def detect_fn(image, detection_model):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections

def load_image_into_numpy_array(path):
    return np.array(Image.open(path))

def absoluteFilePaths(directory):
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))

if __name__ == "__main__":
    main()
