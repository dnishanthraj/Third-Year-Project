### NEED TO FIX: See why not syncing epochs with folds, and why fold lengths vary each fold. Fix!
import pandas as pd
import argparse
import os
from collections import OrderedDict
from glob import glob
import yaml

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import StepLR #Added this
from datetime import datetime
import uuid

# cudnn.benchmark = True
# cudnn.deterministic = False

from sklearn.model_selection import KFold

import albumentations as albu
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from losses import BCEDiceLoss
from dataset import MyLidcDataset
from metrics import iou_score,dice_coef
from utils import AverageMeter, str2bool

from Unet.unet_model import UNet
from UnetNested.Nested_Unet import NestedUNet


# torch.backends.cudnn.enabled = False
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--name', default="UNET",
                        help='model name: UNET',choices=['UNET', 'NestedUNET'])
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=15, type=int, #Changed default to 4 from 12
                        metavar='N', help='mini-batch size (default: 6)')
    parser.add_argument('--early_stopping', default=20, type=int,
                        metavar='N', help='early stopping (default: 50)')
    parser.add_argument('--num_workers', default=12, type=int)

    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float, #Changed to e-3
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # data
    parser.add_argument('--augmentation',type=str2bool,default=False,choices=[True,False])



    config = parser.parse_args()

    return config


def train(train_loader, model, criterion, optimizer):

    
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))

    for input, target in train_loader:

        input = input.cuda()
        target = target.cuda()
        output = model(input)

        loss = criterion(output, target)
        iou = iou_score(output, target)
        dice = dice_coef(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))
        avg_meters['dice'].update(dice, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
            ('dice',avg_meters['dice'].avg)
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice',avg_meters['dice'].avg)])

def validate(val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target in val_loader:
            input = input.cuda().float()
            target = target.cuda().float()

            output = model(input)
            loss = criterion(output, target)
            iou = iou_score(output, target)
            dice = dice_coef(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice',avg_meters['dice'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice',avg_meters['dice'].avg)])

def main():
    # Get configuration
    config = vars(parse_args())
    # torch.backends.cudnn.enabled = False
    # torch.backends.cuda.matmul.allow_tf32 = False

    print("CUDA Available:", torch.cuda.is_available())
    print("CUDA Version:", torch.version.cuda)
    print("cuDNN Version:", torch.backends.cudnn.version())
    print("Allocated Memory:", torch.cuda.memory_allocated())
    print("Reserved Memory:", torch.cuda.memory_reserved())

    # Make Model output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]

    # file_name = f"{config['name']}_{timestamp}_{unique_id}"


    if config['augmentation']== True:
        file_name= f"{config['name']}_with_augmentation_{timestamp}_{unique_id}" 
    else:
        file_name = f"{config['name']}_base_{timestamp}_{unique_id}"
    os.makedirs('model_outputs/{}'.format(file_name),exist_ok=True)
    print("Creating directory called",file_name)

    print('-' * 20)
    print("Configuration Setting as follow")
    for key in config:
        print('{}: {}'.format(key, config[key]))
    print('-' * 20)

    #save configuration
    with open('model_outputs/{}/config.yml'.format(file_name), 'w') as f:
        yaml.dump(config, f)

    #criterion = nn.BCEWithLogitsLoss().cuda()
    criterion = BCEDiceLoss().cuda()
    cudnn.benchmark = True  # Was False - change to True?
    # cudnn.deterministic = True

    # create model
    print("=> creating model" )
    if config['name']=='NestedUNET':
        model = NestedUNet(num_classes=1)
    else:
        model = UNet(n_channels=1, n_classes=1, bilinear=True)
    model = model.cuda()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    params = filter(lambda p: p.requires_grad, model.parameters())


    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    # ADD LEARNING RATE SCHEDULER HERE
    scheduler = StepLR(optimizer, step_size=40, gamma=0.8)  # Reducing LR by 10% every 50 epochs

    # Directory of Image, Mask folder generated from the preprocessing stage ###
    # Write your own directory                                                 #
    IMAGE_DIR = '/dcs/22/u2202609/year_3/cs310/Project/Preprocessing/data/Image/'                                       #
    MASK_DIR = '/dcs/22/u2202609/year_3/cs310/Project/Preprocessing/data/Mask/'      
                                       #
    #Meta Information   - going to implement k-fold here                                                       #
    meta = pd.read_csv('/dcs/22/u2202609/year_3/cs310/Project/Preprocessing/csv/meta.csv')                    #
    ############################################################################
    # Get train/test label from meta.csv
    meta['original_image']= meta['original_image'].apply(lambda x:IMAGE_DIR+ x +'.npy')
    meta['mask_image'] = meta['mask_image'].apply(lambda x:MASK_DIR+ x +'.npy')

    all_meta = meta[meta['data_split'].isin(['Train', 'Validation'])].reset_index(drop=True)

    # # Get all *npy images into list for Train
    # train_image_paths = list(train_meta['original_image'])
    # train_mask_paths = list(train_meta['mask_image'])

    # # Get all *npy images into list for Validation
    # val_image_paths = list(val_meta['original_image'])
    # val_mask_paths = list(val_meta['mask_image']


    # Prepare the dataset for k-fold (all data combined)
    all_image_paths = list(all_meta['original_image'])
    all_mask_paths = list(all_meta['mask_image'])

    
    # Set up 5-Fold Cross Validation
    kf = KFold(n_splits=3, shuffle=True, random_state=26)

    # Create a DataFrame to store averaged results across folds
    log = pd.DataFrame(columns=['epoch', 'lr', 'loss', 'iou', 'dice', 'val_loss', 'val_iou'])

    # Initialize the metric storage
    best_dice = 0
    trigger = 0

    for idx, epoch in enumerate(range(config['epochs'])):
    # Initialize K-Fold
        print(f"\nEpoch {idx}:")
        # Initialize metric accumulators for averaging across folds
        epoch_metrics = {'loss': [], 'iou': [], 'dice': [], 'val_loss': [], 'val_iou': [], 'val_dice': []}
        
        # K-Fold Loop: Each fold gets a separate train/validation split per epoch
        for fold, (train_idx, val_idx) in enumerate(kf.split(all_image_paths)):

            print(f"Starting Fold {fold + 1} of {kf.get_n_splits()}")
        
            # Prepare train/validation sets for this fold
            train_image_paths = [all_image_paths[i] for i in train_idx]
            train_mask_paths = [all_mask_paths[i] for i in train_idx]
            val_image_paths = [all_image_paths[i] for i in val_idx]
            val_mask_paths = [all_mask_paths[i] for i in val_idx]

            # Create Dataset and Dataloaders for this fold
            train_dataset = MyLidcDataset(train_image_paths, train_mask_paths, config['augmentation'])
            val_dataset = MyLidcDataset(val_image_paths, val_mask_paths, config['augmentation'])

            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=config['batch_size'],
                shuffle=True,
                pin_memory=True,
                drop_last=True,
                num_workers=12) # Changed num workers from 6.
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=config['batch_size'],
                shuffle=False,
                pin_memory=True,
                drop_last=False,
                num_workers=12)

            # Train and validate for this fold
            train_log = train(train_loader, model, criterion, optimizer)
            val_log = validate(val_loader, model, criterion)

            # Accumulate fold metrics for averaging after all folds are complete
            epoch_metrics['loss'].append(train_log['loss'])
            epoch_metrics['iou'].append(train_log['iou'])
            epoch_metrics['dice'].append(train_log['dice'])
            epoch_metrics['val_loss'].append(val_log['loss'])
            epoch_metrics['val_iou'].append(val_log['iou'])
            epoch_metrics['val_dice'].append(val_log['dice'])

            # # Accumulate fold metrics for averaging after all folds are complete
            # for key in epoch_metrics:
            #     epoch_metrics[key].append(train_log[key] if 'val' not in key else val_log[key])


        # Average the metrics across all folds after the epoch completes
        avg_metrics = {metric: sum(values) / len(values) for metric, values in epoch_metrics.items()}

        print(f'Epoch [{epoch + 1}/{config["epochs"]}] - Avg Loss: {avg_metrics["loss"]:.4f}, Avg Dice: {avg_metrics["dice"]:.4f}, Avg IOU: {avg_metrics["iou"]:.4f}')
            
        # print("*"*50)
        # print("The lenght of image: {}, mask folders: {} for train".format(len(train_image_paths),len(train_mask_paths)))
        # print("The lenght of image: {}, mask folders: {} for validation".format(len(val_image_paths),len(val_mask_paths)))
        # print("Ratio between Val/ Train is {:2f}".format(len(val_image_paths)/len(train_image_paths)))
        # print("*"*50)
    
    
        # Logging the averaged results
        log = pd.concat([log, pd.DataFrame([{
            'epoch': epoch + 1,
            'lr': config['lr'],
            'loss': avg_metrics['loss'],
            'iou': avg_metrics['iou'],
            'dice': avg_metrics['dice'],
            'val_loss': avg_metrics['val_loss'],
            'val_iou': avg_metrics['val_iou'],
            'val_dice': avg_metrics['val_dice']
        }])], ignore_index=True)

        log_filename = f'model_outputs/{file_name}/log.csv'

        # Save the log file with a unique name
        log.to_csv(log_filename, index=False)


    # log= pd.DataFrame(index=[],columns= ['epoch','lr','loss','iou','dice','val_loss','val_iou'])

        # Model saving logic
        if avg_metrics['val_dice'] > best_dice:
            model_filename = f'model_outputs/{file_name}/model.pth'
            torch.save(model.state_dict(), model_filename)
            best_dice = avg_metrics['val_dice']
            trigger = 0
            print("=> Best model saved (Dice improved)")
        else:
            trigger += 1

        # Early stopping check
        if trigger >= config['early_stopping']:
            print("=> Early stopping triggered")
            break

        scheduler.step()
        torch.cuda.empty_cache()

        # tmp = pd.Series([
        #     epoch,
        #     config['lr'],
        #     train_log['loss'],
        #     train_log['iou'],
        #     train_log['dice'],
        #     val_log['loss'],
        #     val_log['iou'],
        #     val_log['dice']
        # ], index=['epoch', 'lr', 'loss', 'iou', 'dice', 'val_loss', 'val_iou','val_dice'])

        # # log = log.append(tmp, ignore_index=True)
        # log = pd.concat([log, pd.DataFrame([tmp])], ignore_index=True)

if __name__ == '__main__':
    main()
