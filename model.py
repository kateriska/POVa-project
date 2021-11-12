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
        self.create_label_map()

        '''
        python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=train.record

        flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
        flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
        flags.DEFINE_string('image_dir', '', 'Path to images')
        '''
        

    def create_label_map(self):
        labels = [{'name':'warning', 'id':1}, {'name':'complementary', 'id':2}, {'name':'other', 'id':3}, {'name':'information', 'id':4}, {'name':'regulatory', 'id':5}]

        label_map_path = os.path.join('annotations', 'label_map.pbtxt')

        open(label_map_path, 'w+').close()

        with open(label_map_path, 'a') as f:
            for label in labels:
                f.write("item {" + '\n')
                f.write("\t" + "name:'" + label.get('name') + "'" + "\n")
                f.write("\t" + "id:" + str(label.get('id')) + "\n")
                f.write("}" + "\n")
        f.close()
