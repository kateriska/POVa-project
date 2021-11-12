import os
import subprocess
import wget

class DetectionModel:

    def __init__(self, steps):
        super().__init__()

        self.steps = steps

        # these paths set because I use different disc for storing mapillary dataset, annotations and tf records
        self.train_dataset_images_path = "/media/katerina/DATA/mapillaryDataset/mapillary_train0/images"
        self.val_dataset_images_path = "/media/katerina/DATA/mapillaryDataset/mapillary_val/images"

        # tf records could be some GBs for this large dataset
        self.tf_records_train_output = "/media/katerina/DATA/mapillaryDataset/annotations_tf_records/train.record"
        self.tf_records_val_output = "/media/katerina/DATA/mapillaryDataset/annotations_tf_records/val.record"


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
        self.generate_tfrecords()


    # generated TF records of huge Mapilary dataset could be quite large, so keep in mind it during writing to disc, it could take some time
    # but it is basically as large as original archives of images
    # size of TF record for val dataset : 4.6 GB
    # size of TF record for train0 dataset : 10.3 GB
    # each TF record will be used for config pipeline of detection model - than we dont need to load all images anymore, just tf record, because tf record has serialized images and also their annotations
    def generate_tfrecords(self):
        generate_tfrecord_train_command = "python {} --csv_input={} --label_map_path={} --image_dir={} --output_path={}".format(os.path.join('scripts', 'generate_tfrecord.py'), os.path.join('annotations', 'train_annotations.csv'),os.path.join('annotations', 'label_map.pbtxt'),self.train_dataset_images_path,self.tf_records_train_output)
        generate_tfrecord_val_command = "python {} --csv_input={} --label_map_path={} --image_dir={} --output_path={}".format(os.path.join('scripts', 'generate_tfrecord.py'), os.path.join('annotations', 'val_annotations.csv'),os.path.join('annotations', 'label_map.pbtxt'),self.val_dataset_images_path,self.tf_records_val_output)

        subprocess.call(generate_tfrecord_train_command, shell=True)
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
