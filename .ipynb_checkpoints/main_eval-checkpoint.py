import os
import torch
import string
import random
import datetime
import numpy as np
from utils.utils import read_config
from torchvision import transforms
from models.timesformerclipinitvideoguide import (
    TimeSformerCLIPInitVideoGuide,
)
from transformers import TimesformerModel, CLIPTokenizer, CLIPTextModel, CLIPVisionModel, logging
import torchvision
from PIL import Image
import glob
import warnings
warnings.filterwarnings("ignore")



# Set the PYTORCH_CUDA_ALLOC_CONF environment variable
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:50'


def main(args):
    if(args.seed>=0):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        #print("[INFO] Setting SEED: " + str(args.seed))   
    #else:
        #print("[INFO] Setting SEED: None")

    if(torch.cuda.is_available() == False): print("[WARNING] CUDA is not available.")

    #print("[INFO] Found", str(torch.cuda.device_count()), "GPU(s) available.", flush=True)
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")   
    #print("[INFO] Device type:", str(device), flush=True)

    config = read_config()
    if args.dataset == "animalkingdom":
        dataset = 'AnimalKingdom'
    path_data = os.path.join(config['path_dataset'], dataset)

    from datasets.datamanager import DataManager
    manager = DataManager(args, path_data)
    class_list = list(manager.get_act_dict().keys())
    num_classes = len(class_list)

    # training data
    train_transform = manager.get_train_transforms()
    train_loader = manager.get_train_loader(train_transform)


    # val or test data
    val_transform = manager.get_test_transforms()
    val_loader = manager.get_test_loader(val_transform)
#

    # criterion or loss
    import torch.nn as nn
    if args.dataset in ['animalkingdom', 'charades', 'hockey', 'volleyball']:
        criterion = nn.BCEWithLogitsLoss()


    # evaluation metric
    if args.dataset in ['animalkingdom', 'charades']:
        from torchmetrics.classification import MultilabelAveragePrecision
        eval_metric = MultilabelAveragePrecision(num_labels=num_classes, average='micro')
        eval_metric_string = 'Multilabel Average Precision'

    # model
    model_args = (train_loader, val_loader, criterion, eval_metric, class_list, args.test_every, args.distributed, device)
    
    if args.model == 'timesformerclipinitvideoguide':
        from models.timesformerclipinitvideoguide import TimeSformerCLIPInitVideoGuideExecutor

        executor = TimeSformerCLIPInitVideoGuideExecutor(*model_args)
    
    executor.model.to(device)
    
    weights = args.weights    #"/AnimalAI/Weights/checkpoint_Bird.pth"
    weights = '../'+weights
    
    def sample_indices(num_frames):
        if num_frames <= 16:
            indices = np.linspace(0, num_frames - 1, 16, dtype=int)
        else:
            ticks = np.linspace(0, num_frames, 16 + 1, dtype=int)
            indices = ticks[:-1] + (ticks[1:] - ticks[:-1]) // 2
        return indices
    
    transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images if they are not the same size
    transforms.ToTensor(),  # Convert to tensor
    ])
    
    filepath=args.filepath
    image_paths = glob.glob(filepath + '*.jpg')
    num_frames = len(image_paths)
    
    # Sample indices for the frames you want to load
    indices = sample_indices(num_frames)
    # Sort the image paths to ensure consistent order
    sorted_image_paths = sorted(image_paths)

  
    image_list = []
#     print(filepath,weights)
    for idx in indices:
        if idx < len(sorted_image_paths):  #
            filename = sorted_image_paths[idx]
            im = Image.open(filename).convert('RGB')  # Convert to RGB to avoid issues with inconsistent channels
            im = transform(im)  # Apply the transformation
            image_list.append(im)
#     print("image done ", len(image_list))

    image_tensor = torch.stack(image_list, dim=0).to(device)
    image_tensor = image_tensor.unsqueeze(0)
    
    out = executor.predict(image_tensor, weights)
#     print(out.size)
    return out


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Training script for action recognition")
    parser.add_argument("--seed", default=1, type=int, help="Seed for Numpy and PyTorch. Default: -1 (None)")
    parser.add_argument("--epoch_start", default=0, type=int, help="Epoch to start learning from, used when resuming")
    parser.add_argument("--epochs", default=100, type=int, help="Total number of epochs")
    parser.add_argument("--dataset", default="animalkingdom", help="Dataset: volleyball, hockey, charades, ava, animalkingdom")
    parser.add_argument("--model", default="convit", help="Model: convit, query2label")
    parser.add_argument("--total_length", default=16, type=int, help="Number of frames in a video")
    parser.add_argument("--batch_size", default=8, type=int, help="Size of the mini-batch")
    parser.add_argument("--id", default="", help="Additional string appended when saving the checkpoints")
    parser.add_argument("--checkpoint", default="", help="location of a checkpoint file, used to resume training")
    parser.add_argument("--num_workers", default=4, type=int, help="Number of torchvision workers used to load data (default: 8)")
    parser.add_argument("--test_every", default=5, type=int, help="Test the model every this number of epochs")
    parser.add_argument("--gpu", default="0", type=str, help="GPU id in case of multiple GPUs")
    parser.add_argument("--distributed", default=False, type=bool, help="Distributed training flag")
    parser.add_argument("--test_part", default=6, type=int, help="Test partition for Hockey dataset")
    parser.add_argument("--zero_shot", default=False, type=bool, help="Zero-shot or Fully supervised")
    parser.add_argument("--split", default=1, type=int, help="Split 1: 50:50, Split 2: 75:25")
    parser.add_argument("--train", default=True, type=bool, help="train or test")
    parser.add_argument("--animal", default="", help="Animal subset of data to use.")
    parser.add_argument("--filepath", default="")
    parser.add_argument("--videoname",default="")
    parser.add_argument("--weights",default="")
    args = parser.parse_args()
    
    print(main(args))


