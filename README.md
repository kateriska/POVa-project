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
`cd model/research && protoc object_detection/protos/*.proto --python_out=. && cp object_detection/packages/tf2/setup.py . && python -m pip install .`
