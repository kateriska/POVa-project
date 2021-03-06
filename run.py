import argparse
from model import DetectionModel

def parse_args():
    argParser = argparse.ArgumentParser(description='Train model for detection and classification of traffic signs')
    argParser.add_argument('--steps', dest='steps', default=20000, action='store', type=int, help='Number of train steps to train for.')
    argParser.add_argument('--model', dest='model', action='store', default="faster_rcnn", choices=["faster_rcnn", "ssd"], help='Select type of model.')
    return argParser.parse_args()

if __name__ == "__main__":
    init_args = {}
    args = parse_args()
    for arg in vars(args):
        if getattr(args, arg) is not None:
            init_args[arg] = getattr(args, arg)

    #print(init_args)
    model = DetectionModel(**init_args)

    # train model
    model.model_configuration()
