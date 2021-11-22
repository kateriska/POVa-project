# POVa-project

### Setup:

Install virtual environment and activate:  <br />
`python -m venv POVa-project-env` <br />
`source POVa-project-env/bin/activate` <br />

Install libraries: <br />
`pip install opencv-python`  <br />
`pip install tensorflow` <br />
`pip install matplotlib` <br />
`pip install pyyaml` <br />

Clone TensorFlow models: <br />
`git clone https://github.com/tensorflow/models`

Install Tensorflow Object Detection API: <br />
`apt-get install protobuf-compiler` <br />
`cd models/research && protoc object_detection/protos/*.proto --python_out=. && cp object_detection/packages/tf2/setup.py . && python -m pip install .`

Generate TF Records, configure pipeline and prepare for training: <br />
`python run.py`

Train model: <br />
`python models/research/object_detection/model_main_tf2.py --model_dir=trained_model/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8 --pipeline_config_path=trained_model/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8/pipeline.config --num_train_steps=20000`

Run model on test images in ./data/predict folder:
`python predict.py`
