from PIL import Image
import glob
import os
import random
# additional script for converting ppm images to png images for TestIJCNN2013 dataset

# german dataset for test trained model because traffic signs are quite simmilar to Czech Republic and we dont have any huge Czech traffic signs dataset
# https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/published-archive.html - TestIJCNN2013.zip
german_test_dataset_path_ppm = "../data/TestIJCNN2013/TestIJCNN2013Download"
german_test_dataset_path_png = "../data/predict"

converted_dataset_amount = 0.5 # I will only use half of this dataset (50 %) - 150 imgs to try to predict traffic signs - this is not important, you can convert whole 300 dataset to png and predict

for file in glob.glob(german_test_dataset_path_ppm + "/*"):
    extension = os.path.splitext(file)[1][1:]
    if extension != "ppm": # skip Readme which is in downloaded folder along with images
        continue

    if random.random() >= converted_dataset_amount:
        continue

    ppm_img = Image.open(file)

    file_substr = file.split('/')[-1]
    file_substr = file_substr[:len(file_substr)-4]
    print("Converting ppm image: " + file_substr)
    ppm_img.save(german_test_dataset_path_png + "/" + file_substr + ".png")
