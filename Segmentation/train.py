
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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

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
cudnn.benchmark = True

# Set the number of threads dynamically based on available CPU cores
num_threads = torch.get_num_threads()  # Or manually set a value, e.g., num_threads = 8
torch.set_num_threads(num_threads)
torch.set_num_interop_threads(num_threads)

# Set environment variables for threading
os.environ["OMP_NUM_THREADS"] = str(num_threads)
os.environ["MKL_NUM_THREADS"] = str(num_threads)

def parse_args():
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--name', default="UNET",
                        help='model name: UNET',choices=['UNET', 'NestedUNET'])
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=8, type=int, #Changed default to 4 from 12
                        metavar='N', help='mini-batch size (default: 6)')
    parser.add_argument('--early_stopping', default=10, type=int,
                        metavar='N', help='early stopping (default: 50)')
    parser.add_argument('--num_workers', default=12, type=int)

    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float, 
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, #Not needed, sticking with Adam, screw SGD
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

    # pbar = tqdm(total=len(train_loader), desc=f"Rank {dist.get_rank()} Training")
    pbar = tqdm(total=len(train_loader), desc="Training", disable=(dist.get_rank() != 0))


    for input, target in train_loader:

        input = input.cuda()
        target = target.cuda()
        output = model(input)

        loss = criterion(output, target)
        iou = iou_score(output, target)
        dice = dice_coef(output, target)
        
        dist.barrier()  # Ensure all ranks reach this point
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
        # pbar = tqdm(total=len(val_loader), desc=f"Rank {dist.get_rank()} Validation")
        pbar = tqdm(total=len(val_loader), desc="Validation", disable=(dist.get_rank() != 0))

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

def save_checkpoint(state, filename):
    if dist.get_rank() == 0:
        torch.save(state, filename)

def load_checkpoint(filename, model, optimizer, scheduler):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'], checkpoint['best_dice']

def setup_ddp():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    return device

def cleanup_ddp():
    dist.destroy_process_group()

def reduce_mean(tensor):
    """Averages a tensor across all GPUs."""
    if tensor.is_cuda:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return tensor


def main():
    # Get configuration
    device = setup_ddp()

    if dist.get_rank() == 0:
        print(f"Distributed training initialized with {dist.get_world_size()} processes.")

    config = vars(parse_args())
    # torch.backends.cudnn.enabled = False
    # torch.backends.cuda.matmul.allow_tf32 = False
    if dist.get_rank() == 0:
        print("CUDA Available:", torch.cuda.is_available())
        print("CUDA Version:", torch.version.cuda)
        print("cuDNN Version:", torch.backends.cudnn.version())
        print("Allocated Memory:", torch.cuda.memory_allocated())
        print("Reserved Memory:", torch.cuda.memory_reserved())

    # Make Model output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]


    if config['augmentation']== True:
        file_name= f"{config['name']}_with_augmentation_{timestamp}_{unique_id}" 
    else:
        file_name = f"{config['name']}_base_{timestamp}_{unique_id}"
    

    if dist.get_rank() == 0:
        # if not os.path.exists(f'model_outputs/{file_name}'):
        os.makedirs(f'model_outputs/{file_name}', exist_ok=True)
        print(f"Creating directory: model_outputs/{file_name}")
        # Save configuration once if not already present
        config_path = f'model_outputs/{file_name}/config.yml'
    
        # if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        print("Config file created.")

    dist.barrier()

# Ensure all GPUs wait for folder and config.yml creation
    if dist.get_rank() == 0:
        print("Creating directory called",file_name)
    # dist.barrier()

        print('-' * 20)
        print("Configuration Setting as follow")
        for key in config:
            print('{}: {}'.format(key, config[key]))
        print('-' * 20)

    #criterion = nn.BCEWithLogitsLoss().cuda()
    criterion = BCEDiceLoss().cuda()
    cudnn.benchmark = True  # Was False - change to True?
    # cudnn.deterministic = True

    # Initialize distributed training
    # create model
    if dist.get_rank() == 0:
        print("=> creating model" )

    if config['name']=='NestedUNET':
        model = NestedUNet(num_classes=1)
    else:
        model = UNet(n_channels=1, n_classes=1, bilinear=True)
    # model = model.cuda()
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        if dist.get_rank() == 0:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
        # model = nn.DataParallel(model)
        model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])])

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
    #Meta Information                                                      #
    meta = pd.read_csv('/dcs/22/u2202609/year_3/cs310/Project/Preprocessing/csv/meta.csv')                    #
    ############################################################################
    # Get train/test label from meta.csv
    meta['original_image']= meta['original_image'].apply(lambda x:IMAGE_DIR+ x +'.npy')
    meta['mask_image'] = meta['mask_image'].apply(lambda x:MASK_DIR+ x +'.npy')

    all_meta = meta[meta['data_split'].isin(['Train', 'Validation'])].reset_index(drop=True)

    # Check for existing checkpoint
    checkpoint_filename = f'model_outputs/{file_name}/checkpoint.pth'
    log_filename = f'model_outputs/{file_name}/log.csv'

    # Initialize values for starting epoch and best dice score
    dist.barrier()
    if os.path.exists(checkpoint_filename):
        print(f"=> Loading checkpoint from {checkpoint_filename}")
        start_epoch, best_dice = load_checkpoint(checkpoint_filename, model, optimizer, scheduler)
        log = pd.read_csv(log_filename)  # Load existing log if checkpoint exists
    else:
        start_epoch = 0
        best_dice = 0
        log = pd.DataFrame(columns=['epoch', 'lr', 'loss', 'iou', 'dice', 'val_loss', 'val_iou', 'val_dice'])


    # Prepare the dataset for k-fold (all data combined)
    all_image_paths = list(all_meta['original_image'])
    all_mask_paths = list(all_meta['mask_image'])

    
    # Set up 5-Fold Cross Validation
    kf = KFold(n_splits=9, shuffle=True, random_state=26)

    # Create a DataFrame to store averaged results across folds
    log = pd.DataFrame(columns=['epoch', 'lr', 'loss', 'iou', 'dice', 'val_loss', 'val_iou'])

    # Initialize the metric storage
    best_dice = 0
    trigger = 0

    for idx, epoch in enumerate(range(start_epoch, (config['epochs']))):
    # Initialize K-Fold
        if dist.get_rank() == 0:
            print(f"\nEpoch {idx}:")
        # Initialize metric accumulators for averaging across folds
        epoch_metrics = {
            'loss': torch.zeros(1).cuda(),
            'iou': torch.zeros(1).cuda(),
            'dice': torch.zeros(1).cuda(),
            'val_loss': torch.zeros(1).cuda(),
            'val_iou': torch.zeros(1).cuda(),
            'val_dice': torch.zeros(1).cuda()
        }
        
        fold_count = torch.zeros(1).cuda()  # To count the number of folds processed
        # K-Fold Loop: Each fold gets a separate train/validation split per epoch
        for fold, (train_idx, val_idx) in enumerate(kf.split(all_image_paths)):

            if fold % dist.get_world_size() != dist.get_rank():
                continue
            
            if dist.get_rank() == 0:
                print(f"Starting Fold {fold + 1} of {kf.get_n_splits()}")
        
            # Prepare train/validation sets for this fold
            train_image_paths = [all_image_paths[i] for i in train_idx]
            train_mask_paths = [all_mask_paths[i] for i in train_idx]
            val_image_paths = [all_image_paths[i] for i in val_idx]
            val_mask_paths = [all_mask_paths[i] for i in val_idx]

            # Create Dataset and Dataloaders for this fold
            train_dataset = MyLidcDataset(train_image_paths, train_mask_paths, config['augmentation'])
            val_dataset = MyLidcDataset(val_image_paths, val_mask_paths, config['augmentation'])

            # Use DistributedSampler
            train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
            val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)


                    # Create Dataloaders
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=config['batch_size'],
                sampler=train_sampler,  # DistributedSampler here
                pin_memory=True,
                drop_last=True,
                num_workers=config['num_workers']
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=config['batch_size'],
                sampler=val_sampler,  # DistributedSampler here
                pin_memory=True,
                drop_last=False,
                num_workers=config['num_workers']
            )

            # if dist.get_rank() == 0:
            #     print(f"Rank {dist.get_rank()} has {len(train_sampler)} training samples and {len(val_sampler)} validation samples.")


            # Train and validate for this fold
            train_log = train(train_loader, model, criterion, optimizer)
            for key in train_log.keys():
                train_log[key] = reduce_mean(torch.tensor(train_log[key]).cuda()).item()

            val_log = validate(val_loader, model, criterion)

           
            for key in val_log.keys():
                val_log[key] = reduce_mean(torch.tensor(val_log[key]).cuda()).item()

            # Accumulate fold metrics for averaging
            epoch_metrics['loss'] += torch.tensor(train_log['loss']).cuda()
            epoch_metrics['iou'] += torch.tensor(train_log['iou']).cuda()
            epoch_metrics['dice'] += torch.tensor(train_log['dice']).cuda()
            epoch_metrics['val_loss'] += torch.tensor(val_log['loss']).cuda()
            epoch_metrics['val_iou'] += torch.tensor(val_log['iou']).cuda()
            epoch_metrics['val_dice'] += torch.tensor(val_log['dice']).cuda()
            fold_count += 1  # Increment fold count

            print(f"Rank {dist.get_rank()} finished Fold {fold + 1}")

            dist.barrier()


        # print(dist.get_rank(), f"Rank {dist.get_rank()} completed all folds for Epoch {epoch}")

        dist.barrier()
     
        # print(dist.get_rank(), f"Rank {dist.get_rank()} proceeding to the next epoch")

        #Synchronize metrics across all GPUs
        for key in epoch_metrics.keys():
            dist.all_reduce(epoch_metrics[key], op=dist.ReduceOp.SUM)
        dist.all_reduce(fold_count, op=dist.ReduceOp.SUM)

        avg_metrics = {key: (epoch_metrics[key] / fold_count).item() for key in epoch_metrics.keys()}

        if dist.get_rank() == 0:
            print(f'Epoch [{epoch + 1}/{config["epochs"]}] - Avg Training Loss: {avg_metrics["loss"]:.4f}, Avg Training Dice: {avg_metrics["dice"]:.4f}, Avg Training IOU: {avg_metrics["iou"]:.4f}')
        
        # print("*"*50)
        # print("The lenght of image: {}, mask folders: {} for train".format(len(train_image_paths),len(train_mask_paths)))
        # print("The lenght of image: {}, mask folders: {} for validation".format(len(val_image_paths),len(val_mask_paths)))
        # print("Ratio between Val/ Train is {:2f}".format(len(val_image_paths)/len(train_image_paths)))
        # print("*"*50)

        dist.barrier()

        if dist.get_rank() == 0:
            # Load existing log if checkpoint exists
            if os.path.exists(log_filename):
                log = pd.read_csv(log_filename)
                last_logged_epoch = log['epoch'].max()  # Get the last logged epoch
            else:
                log = pd.DataFrame(columns=['epoch', 'lr', 'loss', 'iou', 'dice', 'val_loss', 'val_iou', 'val_dice'])
                last_logged_epoch = 0

        # Inside the epoch loop
        if dist.get_rank() == 0:
            # Only log new epochs to avoid duplication
            if epoch + 1 > last_logged_epoch:
                new_row = {
                    'epoch': epoch + 1,
                    'lr': config['lr'],
                    'loss': avg_metrics['loss'],
                    'iou': avg_metrics['iou'],
                    'dice': avg_metrics['dice'],
                    'val_loss': avg_metrics['val_loss'],
                    'val_iou': avg_metrics['val_iou'],
                    'val_dice': avg_metrics['val_dice']
            }

            # Append the new row directly to the log DataFrame
            log = pd.concat([log, pd.DataFrame([new_row])], ignore_index=True)

            # Write the updated log to the CSV file
            log.to_csv(log_filename, index=False, mode='w', header=True)  # Overwrite to avoid duplication
            print(f"Epoch {epoch + 1} logged successfully.")

        dist.barrier()
        # Model saving logic
        if avg_metrics['val_dice'] > best_dice:
            if dist.get_rank() == 0:
                model_filename = f'model_outputs/{file_name}/model.pth'
                torch.save(model.state_dict(), model_filename)
            best_dice = avg_metrics['val_dice']
            trigger = 0
            if dist.get_rank() == 0:
                print("=> Best model saved (Dice improved)")
        else:
            trigger += 1

        #Save a checkpoint regardless of whether it's the best model
        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_dice': best_dice
        }, checkpoint_filename)

        # Early stopping check
        if trigger >= config['early_stopping']:
            if dist.get_rank() == 0:
                print("=> Early stopping triggered")
            break

        scheduler.step()

        torch.cuda.empty_cache()
    
    cleanup_ddp()

if __name__ == '__main__':
    main()
