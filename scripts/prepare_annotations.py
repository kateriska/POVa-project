import glob
import numpy as np
import os
import json
import csv

# additional script to filter unused Mapillary annotations (which are not in train0, val or test dataset) - will be helpful when transforming to csv files and then to TF records
# and transform of json files to csv files of train, test or val dataset - they will be then transofrmed to TF records via https://github.com/datitran/raccoon_dataset/blob/master/generate_tfrecord.py

# dictionaries of classes for transform only to 5 origin classes - warning, information etc which will be used by our detection model
split_warning_dict = {
    "warning--roadworks--g1": 0,
    "warning--domestic-animals--g3": 1,
    "warning--wild-animals--g1": 2,
    "warning--traffic-signals--g4": 3,
    "warning--loop-270-degree--g1": 4,
    "warning--accidental-area-unsure--g2": 5,
    "warning--texts--g2": 6,
    "warning--pedestrians-crossing--g10": 7,
    "warning--double-reverse-curve-right--g1": 8,
    "warning--hairpin-curve-right--g1": 9,
    "warning--wombat-crossing--g1": 10,
    "warning--curve-right--g2": 11,
    "warning--railroad-crossing-without-barriers--g3": 12,
    "warning--dual-lanes-right-turn-or-go-straight--g1": 13,
    "warning--height-restriction--g2": 14,
    "warning--railroad-crossing-with-barriers--g4": 15,
    "warning--junction-with-a-side-road-perpendicular-left--g1": 16,
    "warning--wild-animals--g4": 17,
    "warning--playground--g1": 18,
    "warning--roadworks--g2": 19,
    "warning--other-danger--g1": 20,
    "warning--t-roads--g2": 21,
    "warning--hairpin-curve-right--g4": 22,
    "warning--pedestrians-crossing--g12": 23,
    "warning--curve-right--g1": 24,
    "warning--traffic-merges-left--g1": 25,
    "warning--restricted-zone--g1": 26,
    "warning--two-way-traffic--g2": 27,
    "warning--horizontal-alignment-right--g1": 28,
    "warning--road-narrows-left--g2": 29,
    "warning--added-lane-right--g1": 30,
    "warning--trucks-crossing--g1": 31,
    "warning--turn-right--g1": 32,
    "warning--pedestrians-crossing--g5": 33,
    "warning--texts--g3": 34,
    "warning--slippery-road-surface--g1": 35,
    "warning--railroad-crossing-without-barriers--g4": 36,
    "warning--pass-left-or-right--g1": 37,
    "warning--bus-stop-ahead--g3": 38,
    "warning--pedestrian-stumble-train--g1": 39,
    "warning--offset-roads--g3": 40,
    "warning--roadworks--g6": 41,
    "warning--junction-with-a-side-road-acute-right--g1": 42,
    "warning--crossroads--g1": 43,
    "warning--bicycles-crossing--g2": 44,
    "warning--divided-highway-ends--g1": 45,
    "warning--junction-with-a-side-road-perpendicular-left--g4": 46,
    "warning--crossroads-with-priority-to-the-right--g1": 47,
    "warning--railroad-crossing-with-barriers--g1": 48,
    "warning--junction-with-a-side-road-perpendicular-right--g1": 49,
    "warning--roadworks--g3": 50,
    "warning--curve-left--g1": 51,
    "warning--trail-crossing--g2": 52,
    "warning--uneven-road--g2": 53,
    "warning--roundabout--g25": 54,
    "warning--railroad-intersection--g4": 55,
    "warning--divided-highway-ends--g2": 56,
    "warning--kangaloo-crossing--g1": 57,
    "warning--uneven-road--g6": 58,
    "warning--roundabout--g1": 59,
    "warning--falling-rocks-or-debris-right--g1": 60,
    "warning--narrow-bridge--g1": 61,
    "warning--double-curve-first-left--g1": 62,
    "warning--winding-road-first-left--g2": 63,
    "warning--road-widens--g1": 64,
    "warning--equestrians-crossing--g2": 65,
    "warning--double-turn-first-right--g1": 66,
    "warning--hairpin-curve-left--g1": 67,
    "warning--pedestrians-crossing--g11": 68,
    "warning--children--g2": 69,
    "warning--turn-right--g2": 70,
    "warning--road-bump--g1": 71,
    "warning--junction-with-a-side-road-perpendicular-left--g3": 72,
    "warning--roadworks--g4": 73,
    "warning--railroad-crossing-with-barriers--g2": 74,
    "warning--domestic-animals--g1": 75,
    "warning--traffic-signals--g3": 76,
    "warning--bicycles-crossing--g1": 77,
    "warning--traffic-signals--g2": 78,
    "warning--junction-with-a-side-road-acute-left--g1": 79,
    "warning--curve-left--g2": 80,
    "warning--railroad-crossing-without-barriers--g1": 81,
    "warning--playground--g3": 82,
    "warning--winding-road-first-right--g1": 83,
    "warning--trams-crossing--g1": 84,
    "warning--double-curve-first-right--g1": 85,
    "warning--traffic-signals--g1": 86,
    "warning--traffic-merges-right--g1": 87,
    "warning--y-roads--g1": 88,
    "warning--railroad-crossing--g1": 89,
    "warning--winding-road-first-left--g1": 90,
    "warning--bicycles-crossing--g3": 91,
    "warning--two-way-traffic--g1": 92,
    "warning--narrow-bridge--g3": 93,
    "warning--double-curve-first-left--g2": 94,
    "warning--stop-ahead--g9": 95,
    "warning--emergency-vehicles--g1": 96,
    "warning--pedestrians-crossing--g9": 97,
    "warning--falling-rocks-or-debris-right--g2": 98,
    "warning--junction-with-a-side-road-perpendicular-right--g3": 99,
    "warning--shared-lane-motorcycles-bicycles--g1": 100,
    "warning--traffic-merges-left--g2": 101,
    "warning--texts--g1": 102,
    "warning--slippery-road-surface--g2": 103,
    "warning--flaggers-in-road--g1": 104,
    "warning--road-narrows-right--g2": 105,
    "warning--pass-left-or-right--g2": 106,
    "warning--pedestrians-crossing--g4": 107,
    "warning--dip--g2": 108,
    "warning--uneven-roads-ahead--g1": 109,
    "warning--falling-rocks-or-debris-right--g4": 110,
    "warning--road-narrows-right--g1": 111,
    "warning--crossroads--g3": 112,
    "warning--hairpin-curve-left--g3": 113,
    "warning--railroad-intersection--g3": 114,
    "warning--horizontal-alignment-right--g3": 115,
    "warning--turn-left--g1": 116,
    "warning--traffic-merges-right--g2": 117,
    "warning--double-curve-first-right--g2": 118,
    "warning--t-roads--g1": 119,
    "warning--horizontal-alignment-left--g1": 120,
    "warning--road-narrows-left--g1": 121,
    "warning--road-narrows--g1": 122,
    "warning--railroad-crossing--g3": 123,
    "warning--slippery-motorcycles--g1": 124,
    "warning--school-zone--g2": 125,
    "warning--road-narrows--g2": 126,
    "warning--children--g1": 127,
    "warning--road-bump--g2": 128,
    "warning--pedestrians-crossing--g1": 129,
    "warning--other-danger--g3": 130,
    "warning--railroad-crossing--g4": 131,
    "warning--steep-ascent--g7": 132,
    "warning--road-widens-right--g1": 133,
    "warning--winding-road-first-right--g3": 134
}

split_other_dict = {
    "other-sign": 0
}

split_information_dict = {
    "information--end-of-motorway--g1": 0,
    "information--living-street--g1": 1,
    "information--minimum-speed-40--g1": 2,
    "information--parking--g45": 3,
    "information--food--g2": 4,
    "information--end-of-pedestrians-only--g2": 5,
    "information--pedestrians-crossing--g1": 6,
    "information--stairs--g1": 7,
    "information--parking--g3": 8,
    "information--motorway--g1": 9,
    "information--end-of-living-street--g1": 10,
    "information--hospital--g1": 11,
    "information--airport--g2": 12,
    "information--bike-route--g1": 13,
    "information--telephone--g1": 14,
    "information--highway-interstate-route--g2": 15,
    "information--gas-station--g3": 16,
    "information--lodging--g1": 17,
    "information--pedestrians-crossing--g2": 18,
    "information--telephone--g2": 19,
    "information--gas-station--g1": 20,
    "information--bus-stop--g1": 21,
    "information--road-bump--g1": 22,
    "information--end-of-built-up-area--g1": 23,
    "information--interstate-route--g1": 24,
    "information--parking--g5": 25,
    "information--parking--g2": 26,
    "information--parking--g6": 27,
    "information--no-parking--g3": 28,
    "information--safety-area--g2": 29,
    "information--camp--g1": 30,
    "information--airport--g1": 31,
    "information--highway-exit--g1": 32,
    "information--dead-end--g1": 33,
    "information--trailer-camping--g1": 34,
    "information--end-of-limited-access-road--g1": 35,
    "information--dead-end-except-bicycles--g1": 36,
    "information--emergency-facility--g2": 37,
    "information--limited-access-road--g1": 38,
    "information--central-lane--g1": 39,
    "information--tram-bus-stop--g2": 40,
    "information--parking--g1": 41,
    "information--disabled-persons--g1": 42,
    "information--children--g1": 43
}

split_regulatory_dict = {
    "regulatory--maximum-speed-limit-65--g2": 0,
    "regulatory--no-right-turn--g2": 1,
    "regulatory--triple-lanes-turn-left-center-lane--g1": 2,
    "regulatory--maximum-speed-limit-25--g1": 3,
    "regulatory--no-stopping--g5": 4,
    "regulatory--mopeds-and-bicycles-only--g1": 5,
    "regulatory--no-right-turn--g1": 6,
    "regulatory--no-u-turn--g2": 7,
    "regulatory--maximum-speed-limit-50--g6": 8,
    "regulatory--keep-right--g1": 9,
    "regulatory--dual-path-pedestrians-and-bicycles--g1": 10,
    "regulatory--go-straight-or-turn-right--g3": 11,
    "regulatory--turn-left--g1": 12,
    "regulatory--no-buses--g3": 13,
    "regulatory--maximum-speed-limit-led-60--g1": 14,
    "regulatory--stop--g1": 15,
    "regulatory--end-of-buses-only--g1": 16,
    "regulatory--no-overtaking--g4": 17,
    "regulatory--maximum-speed-limit-10--g1": 18,
    "regulatory--passing-lane-ahead--g1": 19,
    "regulatory--maximum-speed-limit-led-100--g1": 20,
    "regulatory--no-heavy-goods-vehicles--g5": 21,
    "regulatory--width-limit--g1": 22,
    "regulatory--no-heavy-goods-vehicles--g2": 23,
    "regulatory--go-straight--g3": 24,
    "regulatory--maximum-speed-limit-60--g1": 25,
    "regulatory--no-stopping--g15": 26,
    "regulatory--one-way-straight--g1": 27,
    "regulatory--maximum-speed-limit-30--g1": 28,
    "regulatory--road-closed--g1": 29,
    "regulatory--one-way-right--g1": 30,
    "regulatory--give-way-to-oncoming-traffic--g1": 31,
    "regulatory--dual-lanes-turn-right-or-straight--g1": 32,
    "regulatory--one-way-left--g3": 33,
    "regulatory--road-closed-to-vehicles--g1": 34,
    "regulatory--maximum-speed-limit-35--g2": 35,
    "regulatory--no-overtaking-by-heavy-goods-vehicles--g1": 36,
    "regulatory--end-of-maximum-speed-limit-70--g1": 37,
    "regulatory--no-parking-or-no-stopping--g2": 38,
    "regulatory--maximum-speed-limit-25--g2": 39,
    "regulatory--u-turn--g1": 40,
    "regulatory--turn-right-ahead--g2": 41,
    "regulatory--no-pedestrians--g2": 42,
    "regulatory--priority-road--g4": 43,
    "regulatory--no-pedestrians--g3": 44,
    "regulatory--weight-limit--g1": 45,
    "regulatory--minimum-safe-distance--g1": 46,
    "regulatory--no-turns--g1": 47,
    "regulatory--turn-left--g3": 48,
    "regulatory--lane-control--g1": 49,
    "regulatory--stop--g2": 50,
    "regulatory--no-turn-on-red--g2": 51,
    "regulatory--maximum-speed-limit-90--g1": 52,
    "regulatory--one-way-right--g2": 53,
    "regulatory--no-entry--g1": 54,
    "regulatory--maximum-speed-limit-30--g3": 55,
    "regulatory--maximum-speed-limit-45--g3": 56,
    "regulatory--maximum-speed-limit-110--g1": 57,
    "regulatory--maximum-speed-limit-20--g1": 58,
    "regulatory--no-right-turn--g3": 59,
    "regulatory--maximum-speed-limit-50--g1": 60,
    "regulatory--end-of-prohibition--g1": 61,
    "regulatory--no-bicycles--g3": 62,
    "regulatory--yield--g1": 63,
    "regulatory--no-left-turn--g1": 64,
    "regulatory--turn-right--g2": 65,
    "regulatory--maximum-speed-limit-55--g2": 66,
    "regulatory--no-left-turn--g3": 67,
    "regulatory--go-straight-or-turn-left--g3": 68,
    "regulatory--dual-lanes-go-straight-on-right--g1": 69,
    "regulatory--end-of-speed-limit-zone--g1": 70,
    "regulatory--no-stopping--g8": 71,
    "regulatory--maximum-speed-limit-15--g1": 72,
    "regulatory--do-not-stop-on-tracks--g1": 73,
    "regulatory--no-turn-on-red--g3": 74,
    "regulatory--no-parking-or-no-stopping--g3": 75,
    "regulatory--dual-lanes-turn-left-or-straight--g1": 76,
    "regulatory--height-limit--g1": 77,
    "regulatory--bicycles-only--g3": 78,
    "regulatory--no-overtaking--g2": 79,
    "regulatory--roundabout--g2": 80,
    "regulatory--shared-path-bicycles-and-pedestrians--g1": 81,
    "regulatory--left-turn-yield-on-green--g1": 82,
    "regulatory--maximum-speed-limit-100--g3": 83,
    "regulatory--bicycles-only--g1": 84,
    "regulatory--no-bicycles--g2": 85,
    "regulatory--wrong-way--g1": 86,
    "regulatory--no-motor-vehicles-except-motorcycles--g1": 87,
    "regulatory--keep-right--g4": 88,
    "regulatory--no-overtaking--g1": 89,
    "regulatory--pedestrians-only--g1": 90,
    "regulatory--dual-lanes-turn-left-no-u-turn--g1": 91,
    "regulatory--no-left-turn--g2": 92,
    "regulatory--maximum-speed-limit-45--g1": 93,
    "regulatory--go-straight-or-turn-right--g1": 94,
    "regulatory--pedestrians-only--g2": 95,
    "regulatory--no-motor-vehicles-except-motorcycles--g2": 96,
    "regulatory--weight-limit-with-trucks--g1": 97,
    "regulatory--no-parking-or-no-stopping--g1": 98,
    "regulatory--no-straight-through--g1": 99,
    "regulatory--pass-on-either-side--g2": 100,
    "regulatory--end-of-maximum-speed-limit-30--g2": 101,
    "regulatory--end-of-maximum-speed-limit-70--g2": 102,
    "regulatory--no-motor-vehicles--g4": 103,
    "regulatory--go-straight-or-turn-left--g1": 104,
    "regulatory--maximum-speed-limit-120--g1": 105,
    "regulatory--end-of-priority-road--g1": 106,
    "regulatory--maximum-speed-limit-led-80--g1": 107,
    "regulatory--keep-right--g2": 108,
    "regulatory--no-motorcycles--g1": 109,
    "regulatory--reversible-lanes--g2": 110,
    "regulatory--road-closed-to-vehicles--g3": 111,
    "regulatory--no-stopping--g2": 112,
    "regulatory--turn-left--g2": 113,
    "regulatory--one-way-left--g1": 114,
    "regulatory--buses-only--g1": 115,
    "regulatory--no-heavy-goods-vehicles-or-buses--g1": 116,
    "regulatory--no-u-turn--g3": 117,
    "regulatory--turn-right--g1": 118,
    "regulatory--go-straight-or-turn-left--g2": 119,
    "regulatory--no-straight-through--g2": 120,
    "regulatory--parking-restrictions--g2": 121,
    "regulatory--stop-here-on-red-or-flashing-light--g2": 122,
    "regulatory--end-of-bicycles-only--g1": 123,
    "regulatory--no-motorcycles--g2": 124,
    "regulatory--no-pedestrians-or-bicycles--g1": 125,
    "regulatory--truck-speed-limit-60--g1": 126,
    "regulatory--no-u-turn--g1": 127,
    "regulatory--one-way-right--g3": 128,
    "regulatory--stop-here-on-red-or-flashing-light--g1": 129,
    "regulatory--roundabout--g1": 130,
    "regulatory--stop-signals--g1": 131,
    "regulatory--maximum-speed-limit-40--g1": 132,
    "regulatory--road-closed--g2": 133,
    "regulatory--maximum-speed-limit-70--g1": 134,
    "regulatory--keep-right--g6": 135,
    "regulatory--stop--g10": 136,
    "regulatory--no-bicycles--g1": 137,
    "regulatory--keep-left--g2": 138,
    "regulatory--turn-right--g3": 139,
    "regulatory--no-turn-on-red--g1": 140,
    "regulatory--keep-left--g1": 141,
    "regulatory--turn-left-ahead--g1": 142,
    "regulatory--turn-right-ahead--g1": 143,
    "regulatory--pass-on-either-side--g1": 144,
    "regulatory--no-heavy-goods-vehicles--g1": 145,
    "regulatory--dual-path-bicycles-and-pedestrians--g1": 146,
    "regulatory--priority-over-oncoming-vehicles--g1": 147,
    "regulatory--text-four-lines--g1": 148,
    "regulatory--maximum-speed-limit-100--g1": 149,
    "regulatory--no-parking--g5": 150,
    "regulatory--end-of-no-parking--g1": 151,
    "regulatory--no-parking--g1": 152,
    "regulatory--no-motor-vehicles--g1": 153,
    "regulatory--go-straight--g1": 154,
    "regulatory--maximum-speed-limit-5--g1": 155,
    "regulatory--no-motor-vehicle-trailers--g1": 156,
    "regulatory--no-heavy-goods-vehicles--g4": 157,
    "regulatory--bicycles-only--g2": 158,
    "regulatory--one-way-left--g2": 159,
    "regulatory--maximum-speed-limit-80--g1": 160,
    "regulatory--no-parking--g2": 161,
    "regulatory--no-mopeds-or-bicycles--g1": 162,
    "regulatory--maximum-speed-limit-40--g6": 163,
    "regulatory--no-pedestrians--g1": 164,
    "regulatory--dual-lanes-go-straight-on-left--g1": 165,
    "regulatory--detour-left--g1": 166,
    "regulatory--maximum-speed-limit-40--g3": 167,
    "regulatory--do-not-block-intersection--g1": 168,
    "regulatory--turning-vehicles-yield-to-pedestrians--g1": 169,
    "regulatory--radar-enforced--g1": 170,
    "regulatory--shared-path-pedestrians-and-bicycles--g1": 171,
    "regulatory--no-stopping--g4": 172,
    "regulatory--no-vehicles-carrying-dangerous-goods--g1": 173,
    "regulatory--no-hawkers--g1": 174,
    "regulatory--no-overtaking--g5": 175
}

split_complementary_dict = {
    "complementary--maximum-speed-limit-30--g1": 0,
    "complementary--go-right--g2": 1,
    "complementary--trucks--g1": 2,
    "complementary--maximum-speed-limit-35--g1": 3,
    "complementary--priority-route-at-intersection--g1": 4,
    "complementary--obstacle-delineator--g2": 5,
    "complementary--chevron-left--g1": 6,
    "complementary--both-directions--g1": 7,
    "complementary--keep-right--g1": 8,
    "complementary--chevron-left--g3": 9,
    "complementary--one-direction-right--g1": 10,
    "complementary--keep-left--g1": 11,
    "complementary--extent-of-prohibition-area-both-direction--g1": 12,
    "complementary--tow-away-zone--g1": 13,
    "complementary--chevron-left--g2": 14,
    "complementary--except-bicycles--g1": 15,
    "complementary--maximum-speed-limit-50--g1": 16,
    "complementary--buses--g1": 17,
    "complementary--maximum-speed-limit-20--g1": 18,
    "complementary--distance--g3": 19,
    "complementary--chevron-right--g3": 20,
    "complementary--maximum-speed-limit-75--g1": 21,
    "complementary--distance--g1": 22,
    "complementary--maximum-speed-limit-45--g1": 23,
    "complementary--chevron-left--g5": 24,
    "complementary--maximum-speed-limit-15--g1": 25,
    "complementary--obstacle-delineator--g1": 26,
    "complementary--accident-area--g3": 27,
    "complementary--distance--g2": 28,
    "complementary--maximum-speed-limit-25--g1": 29,
    "complementary--turn-left--g2": 30,
    "complementary--chevron-right--g4": 31,
    "complementary--chevron-right--g1": 32,
    "complementary--chevron-left--g4": 33,
    "complementary--chevron-right--g5": 34,
    "complementary--maximum-speed-limit-55--g1": 35,
    "complementary--maximum-speed-limit-40--g1": 36,
    "complementary--turn-right--g2": 37,
    "complementary--chevron-right-unsure--g6": 38,
    "complementary--one-direction-left--g1": 39,
    "complementary--maximum-speed-limit-70--g1": 40,
    "complementary--go-left--g1": 41,
    "complementary--trucks-turn-right--g1": 42,
    "complementary--pass-right--g1": 43,
    "complementary--go-right--g1": 44
}

# write to csv used images with their particular folder (val, train or test)
def get_used_image_names(train_dataset_images_path, val_dataset_images_path, test_dataset_images_path):

    f = open("used_data.csv","w+")
    for file in glob.glob(train_dataset_images_path + "/*"):
        file_substr = file.split('/')[-1]
        file_substr = file_substr[:len(file_substr)-4]

        f.write(file_substr + ",train\n")

    for file in glob.glob(val_dataset_images_path + "/*"):
        file_substr = file.split('/')[-1]
        file_substr = file_substr[:len(file_substr)-4]

        f.write(file_substr + ",val\n")

    for file in glob.glob(test_dataset_images_path + "/*"):
        file_substr = file.split('/')[-1]
        file_substr = file_substr[:len(file_substr)-4]

        f.write(file_substr + ",test\n")

    f.close()


# remove of unused annotations (they are not in train0, val or test folder)
def remove_unused_annotations(annotations_path):
    used_data = np.genfromtxt('used_data.csv',delimiter=",", usecols=(0), dtype=None, encoding=None)
    for file in glob.glob(annotations_path + "/*"):
        file_substr = file.split('/')[-1]
        file_substr = file_substr[:len(file_substr)-5]
        if file_substr not in used_data:
            print("Remove this annotation - we dont use this file")
            #os.remove("/media/katerina/DATA/mapillaryDataset/mapillary_annotations/mtsd_v2_fully_annotated/annotations/" + file_substr + ".json")
        else:
            print("Keep annotation")

# convert json to csv and each object is written to new line (see e.g. https://github.com/datitran/raccoon_dataset/blob/master/data/train_labels.csv)
def json_to_csv(file, train_annotations, test_annotations, val_annotations):
    file_substr = file.split('/')[-1]
    file_substr = file_substr[:len(file_substr)-5]

    with open(file) as json_file:
        data = json.load(json_file)

    width = str(data['width'])
    height = str(data['height'])

    with open("used_data.csv", "r") as f:
        reader = csv.reader(f, delimiter=",")
        for line in reader:
            if file_substr == line[0]:
                # figure out whether file is from train, val or test dataset
                folder_type = line[1]
                break

    for object in data['objects']:
        label = edit_labels(object['label'], split_warning_dict, split_complementary_dict, split_other_dict, split_information_dict, split_regulatory_dict)
        xmin = str(object["bbox"]["xmin"])
        ymin = str(object["bbox"]["ymin"])
        xmax = str(object["bbox"]["xmax"])
        ymax = str(object["bbox"]["ymax"])

        annotation = file_substr + ".jpg," + width + "," + height + "," + label + "," + xmin + "," + ymin + "," + xmax + "," + ymax + "\n"

        # write to particular annotation csv
        if folder_type == "train":
            train_annotations.write(annotation)
        elif folder_type == "val":
            val_annotations.write(annotation)
        elif folder_type == "test":
            test_annotations.write(annotation)


# labels are trimmed only to 5 original - it is unecessary to use our detection model with all more than 300 classes
def edit_labels(original_label, split_warning_dict, split_complementary_dict, split_other_dict, split_information_dict, split_regulatory_dict):
    if any(original_label in d for d in split_warning_dict):
        new_label = "warning"
    elif any(original_label in d for d in split_complementary_dict):
        new_label = "complementary"
    elif any(original_label in d for d in split_other_dict):
        new_label = "other"
    elif any(original_label in d for d in split_information_dict):
        new_label = "information"
    elif any(original_label in d for d in split_regulatory_dict):
        new_label = "regulatory"

    return new_label



# get path to images of train0, test and val dataset - these relative paths are here because I use different HDD disc for storing data than my SSD

get_used_image_names("/media/katerina/DATA/mapillaryDataset/mapillary_train0/images", "/media/katerina/DATA/mapillaryDataset/mapillary_val/images", "/media/katerina/DATA/mapillaryDataset/mapillary_test/images")
annotations_path = "/media/katerina/DATA/mapillaryDataset/mapillary_annotations/mtsd_v2_fully_annotated/annotations"

#remove_unused_annotations(annotations_path)
train_annotations = open("../annotations/train_annotations.csv","w+")
test_annotations = open("../annotations/test_annotations.csv","w+")
val_annotations = open("../annotations/val_annotations.csv","w+")

# write header of annotation csvs
train_annotations.write("filename,width,height,class,xmin,ymin,xmax,ymax\n")
test_annotations.write("filename,width,height,class,xmin,ymin,xmax,ymax\n")
val_annotations.write("filename,width,height,class,xmin,ymin,xmax,ymax\n")

for file in glob.glob(annotations_path + "/*"):
    json_to_csv(file, train_annotations, test_annotations, val_annotations)
