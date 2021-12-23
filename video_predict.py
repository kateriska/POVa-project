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

matplotlib.use('TkAgg')

PATH_TO_MODEL_DIR="trained_model/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8"

PATH_TO_CFG = PATH_TO_MODEL_DIR + "/pipeline.config"

PATH_TO_CKPT = PATH_TO_MODEL_DIR #+ "/checkpoint"
CHECKPOINT = "ckpt-26"

PATH_TO_LABELS = "annotations/label_map.pbtxt"

print('Loading model... ', end='')
start_time = time.time()

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(PATH_TO_CKPT, CHECKPOINT)).expect_partial()

@tf.function
def detect_fn(image):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections

end_time = time.time()
elapsed_time = end_time - start_time
print('Model loaded in {} seconds'.format(elapsed_time))


category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

video_file_name = ""

videoCapture = cv2.VideoCapture(video_file_name)    
if not videoCapture.isOpened():
    print("Error: Unable to open video file for reading", video_file_name)
    exit(-1)
    
while videoCapture.isOpened():
    ret, frame = videoCapture.read()
    if not ret:
        break

    image_np = np.array(frame)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    
    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    if num_detections > 0:
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
        
        cv2.imshow(image_np_with_detections)
    else:
        cv2.imshow(image_np)

    if cv2.waitKey(10) == 27:
            break

cv2.destroyAllWindows()
videoCapture.release()