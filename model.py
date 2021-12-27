import os
import subprocess
import wget
import shutil
import tensorflow as tf
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

class DetectionModel:

    def __init__(self, steps, model):
        super().__init__()

        self.steps = steps

        # path to train0 and mapillary_val dataset images - mtsd_fully_annotated_images.train.0.zip and mtsd_fully_annotated_images.val.zip (https://www.mapillary.com/dataset/trafficsign)
        self.train_dataset_images_path = "./mapillary_dataset/train0/images"
        self.val_dataset_images_path = "./mapillary_dataset/mapillary_val/images"

        # tf records could have size some GBs for this large dataset
        self.tf_records_train_output = "./tf_records/train.record"
        self.tf_records_val_output = "./tf_records/val.record"

        # name and download link to used model
        if model == "faster_rcnn":
            self.detection_model_name = 'faster_rcnn_resnet50_v1_640x640_coco17_tpu-8'
            self.detection_model_url = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz'
        elif model == "ssd":
            self.detection_model_name = 'ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8'
            self.detection_model_url = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz'

        self.model = model


    # download pretrained model from TensorFlow Detection Model Zoo
    def download_pretrained_model(self):
        if not os.path.exists(os.path.join('pretrained_model',self.detection_model_name)):
            filename = wget.download(self.detection_model_url)
            archive_name = self.detection_model_name + '.tar.gz'
            os.rename(archive_name, os.path.join('pretrained_model', archive_name))
            os.chdir('./pretrained_model')
            subprocess.call(['tar', '-zxvf', archive_name])
            os.chdir("..")

    def model_configuration(self):
        self.download_pretrained_model()
        self.create_label_map()
        self.generate_tfrecords()
        self.pipeline_configuration()
    
        print("Command for training:")
        # python models/research/object_detection/model_main_tf2.py --model_dir=trained_model/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8 --pipeline_config_path=trained_model/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8/pipeline.config --num_train_steps=20000
        train_command = "python {} --model_dir={} --pipeline_config_path={} --num_train_steps={}".format(os.path.join('models','research','object_detection','model_main_tf2.py'), os.path.join('trained_model', self.detection_model_name), os.path.join('trained_model', self.detection_model_name, 'pipeline.config'), self.steps)
        print(train_command)

        print("Command for launching Tensorboard for metrics:")
        tensorboard_command = "tensorboard --logdir={}".format(os.path.join('trained_model', self.detection_model_name))
        print(tensorboard_command)

    def pipeline_configuration(self):
        if not os.path.exists(os.path.join('trained_model',self.detection_model_name)):
            os.mkdir(os.path.join('trained_model',self.detection_model_name))
        shutil.copyfile(os.path.join('pretrained_model', self.detection_model_name, 'pipeline.config'), os.path.join('trained_model', self.detection_model_name, 'pipeline.config'))

        model_pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
        with tf.io.gfile.GFile(os.path.join('trained_model', self.detection_model_name, 'pipeline.config'), "r") as f:
            text_format.Merge(f.read(), model_pipeline_config)

        if self.model == "faster_rcnn":
            model_pipeline_config.model.faster_rcnn.num_classes = 5 # warning, complementary, other, information, regulatory
        elif self.model == "ssd":
            model_pipeline_config.model.ssd.num_classes = 5 # warning, complementary, other, information, regulatory
        model_pipeline_config.train_config.batch_size = 4

        model_latest_checkpoint = tf.train.latest_checkpoint(os.path.join('trained_model', self.detection_model_name))

        # if our model of detection and classification has some checkpoint, load them and continue training, otherwise load initial checkpoint of downloaded model
        if model_latest_checkpoint is None:
            model_pipeline_config.train_config.fine_tune_checkpoint = os.path.join('pretrained_model', self.detection_model_name, 'checkpoint', 'ckpt-0')
            model_pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
        else:
            # if we continue training fine_tune_checkpoint_type is "full" -  Restores the entire detection model, including the feature extractor, its classification backbone, and the prediction heads
            model_pipeline_config.train_config.fine_tune_checkpoint = model_latest_checkpoint
            model_pipeline_config.train_config.fine_tune_checkpoint_type = "full"

        model_pipeline_config.train_input_reader.label_map_path = os.path.join('annotations', 'label_map.pbtxt')
        model_pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [self.tf_records_train_output]
        model_pipeline_config.eval_input_reader[0].label_map_path = os.path.join('annotations', 'label_map.pbtxt')
        model_pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [self.tf_records_val_output]

        model_pipeline_config_text_format = text_format.MessageToString(model_pipeline_config)
        with tf.io.gfile.GFile(os.path.join('trained_model', self.detection_model_name, 'pipeline.config'), "wb") as f:
            f.write(model_pipeline_config_text_format)


    # generated TF records of huge Mapilary dataset could be quite large, so keep in mind it during writing to disc, it could take some time
    # but it is basically as large as original archives of images
    # size of TF record for val dataset : 4.6 GB
    # size of TF record for train0 dataset : 10.3 GB
    # each TF record will be used for config pipeline of detection model - than we dont need to load all images anymore, just tf record, because tf record has serialized images and also their annotations
    def generate_tfrecords(self):
        generate_tfrecord_train_command = "python {} --csv_input={} --label_map_path={} --image_dir={} --output_path={}".format(os.path.join('scripts', 'generate_tfrecord.py'), os.path.join('annotations', 'train_annotations.csv'),os.path.join('annotations', 'label_map.pbtxt'),self.train_dataset_images_path,self.tf_records_train_output)
        generate_tfrecord_val_command = "python {} --csv_input={} --label_map_path={} --image_dir={} --output_path={}".format(os.path.join('scripts', 'generate_tfrecord.py'), os.path.join('annotations', 'val_annotations.csv'),os.path.join('annotations', 'label_map.pbtxt'),self.val_dataset_images_path,self.tf_records_val_output)
        #subprocess.call(generate_tfrecord_train_command, shell=True)
        #subprocess.call(generate_tfrecord_val_command, shell=True)

    # create map of used labels - 5 original, because we dont need to train on all more than 300 labels
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
