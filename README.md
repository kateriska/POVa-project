# POVa-project
[Github repository](https://github.com/kateriska/POVa-project) <br />
[Colab file](https://colab.research.google.com/drive/1WhBFSG9x85ifHsHqoMQLUBon5PgPnh1_?usp=sharing) <br />
[Drive folder with TF records, trained models and results](https://drive.google.com/drive/folders/1d5bqZQbEn4IcX6OELtsTVnhcKqOzl7_M?usp=sharing) <br />
### Setup:

Install virtual environment and activate:  <br />
`python -m venv POVa-project-env` <br />
`source POVa-project-env/bin/activate` <br />

Install libraries: <br />
`pip install opencv-python`  <br />
`pip install tensorflow` <br />
`pip install matplotlib` <br />
`pip install pyyaml` <br />

Create .gitignore folders hierarchy:  <br />
`mkdir ./pretrained_model` - store of downloaded model from TensorFlow Detection Model ZOO  <br />
`mkdir ./trained_model` - store of configured models for our task, training checkpoints and metrics for Tensorboard are also stored here in folder of each model type<br />
`mkdir ./tf_records` - store of generated TF records  <br />
`mkdir ./trained_model/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8`  <br />
`mkdir ./trained_model/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8`  <br />
`mkdir ./mapillary_dataset` - for store Mapillary dataset <br />
`mkdir ./data/predict` - for store dataset used for predictions <br />
`mkdir ./data/predict/TestIJCNN2013` - folder with ppm images of TestIJCNN2013.zip (German dataset used for trained model prediction - `https://benchmark.ini.rub.de/gtsdb_dataset.html`) <br />
`mkdir ./output_predict` - for store predicted images from  `./data/predict` folder <br />

Download Mapilary dataset (`https://www.mapillary.com/dataset/trafficsign`) to `./mapillary_dataset` folder:  <br />
`mtsd_fully_annotated_annotation.zip`  <br />
`mtsd_fully_annotated_images.test.zip`  <br />
`mtsd_fully_annotated_images.train.0.zip`  <br />
`mtsd_fully_annotated_images.val.zip`  <br />

Path to extracted Mapillary annotations: `./mapillary_dataset/mapillary_annotations/mtsd_v2_fully_annotated/annotations` <br />
Path to extracted Mapillary train, val and test images: `./mapillary_dataset/mapillary_train0/images`, `./mapillary_dataset/mapillary_val/images`, `./mapillary_dataset/mapillary_test/images`

Clone TensorFlow models: <br />
`git clone https://github.com/tensorflow/models`

Install Tensorflow Object Detection API: <br />
`apt-get install protobuf-compiler` <br />
`cd models/research && protoc object_detection/protos/*.proto --python_out=. && cp object_detection/packages/tf2/setup.py . && python -m pip install .`

Generate TF Records, configure pipeline and prepare for training: <br />
`python run.py [--model {"faster_rcnn", "ssd"}] [--steps NUM_TRAIN_STEPS]`

Train model (command is also generated by `run.py` script): <br />
`python models/research/object_detection/model_main_tf2.py --model_dir=trained_model/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8 --pipeline_config_path=trained_model/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8/pipeline.config --num_train_steps=20000`

Launch Tensorboard during training:  <br />
`tensorboard --logdir=trained_model/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8`

Run trained model on a set of images or a video using `predict.py`: <br />
Usage:  <br />
`python predict.py [-h] [-m {faster_rcnn,ssd}] {image,video}`  <br />
`python predict.py image [-h] [-i INPUT_IMAGE_FOLDER] [-o OUTPUT_IMAGE_FOLDER]` <br />
`python predict.py video [-h] [-i INPUT_VIDEO_FILE_NAME] [-o OUTPUT_VIDEO_FILE_NAME]`
