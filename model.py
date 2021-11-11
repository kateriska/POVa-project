import os
import subprocess
import wget

class DetectionModel:

    def __init__(self, steps):
        super().__init__()

        self.steps = steps



    # download pretrained model from TensorFlow Detection Model Zoo
    def download_pretrained_model(self, full_model_name, url):
        if not os.path.exists(os.path.join('pretrained_model','faster_rcnn_resnet50_v1_640x640_coco17_tpu-8')):
            filename = wget.download(url)
            archive_name = full_model_name + '.tar.gz'
            os.rename(archive_name, os.path.join('pretrained_model', archive_name))
            os.chdir('./pretrained_model')
            subprocess.call(['tar', '-zxvf', archive_name])
            os.chdir("..")

    def model_configuration(self):
        self.download_pretrained_model('faster_rcnn_resnet50_v1_640x640_coco17_tpu-8', 'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz')
