
# ---------------------------------------------------------
# Importing required libraries for data processing, training, and evaluation
# ---------------------------------------------------------

import pandas as pd                    # Data manipulation and analysis
import argparse                         # Command-line argument parsing
import os                               # OS operations (path, environment variables)
from collections import OrderedDict     # Maintaining order of dictionary elements
from glob import glob                   # File pattern matching
import yaml                             # YAML file processing (saving/loading configs)
import time                             # Timing utilities

# PyTorch-related imports for model training
import torch
import torch.backends.cudnn as cudnn    # cuDNN backend for GPU acceleration
import torch.nn as nn                   # Neural network components
import torch.optim as optim             # Optimization algorithms
from torch.optim import lr_scheduler    # Learning rate scheduling
from torch.optim.lr_scheduler import StepLR  # Step learning rate decay
from datetime import datetime           # Timestamping
import uuid                              # Unique identifier generation
import torch.distributed as dist         # Distributed training
from torch.nn.parallel import DistributedDataParallel as DDP   # DDP wrapper
from torch.utils.data.distributed import DistributedSampler    # Distributed sampling

# Machine Learning utilities
from sklearn.model_selection import KFold       # K-Fold cross-validation
import albumentations as albu                    # Data augmentation library
from sklearn.model_selection import train_test_split  # Splitting data into train/test
from tqdm import tqdm                             # Progress bar for loops

# Project-specific modules
from losses import BCEDiceLoss, BCEDiceFocalLoss, FocalLoss    # Loss functions
from dataset import MyLidcDataset                              # Custom dataset class
from metrics import iou_score, dice_coef                       # Evaluation metrics
from utils import AverageMeter, str2bool                       # Utility functions

# Model architectures
from Unet.unet_model import UNet                    # Standard U-Net model
from UnetNested.Nested_Unet import NestedUNet        # Nested U-Net model (UNet++)

# ---------------------------------------------------------
# Environment Setup
# ---------------------------------------------------------

# Enable cuDNN auto-tuner for faster training (adjusts algorithms for hardware)
cudnn.benchmark = True

# Dynamically set the number of CPU threads for data loading and operations
num_threads = torch.get_num_threads()
torch.set_num_threads(num_threads)
torch.set_num_interop_threads(num_threads)

# Set environment variables for thread parallelism
os.environ["OMP_NUM_THREADS"] = str(num_threads)
os.environ["MKL_NUM_THREADS"] = str(num_threads)

# ---------------------------------------------------------
# Argument Parsing Function
# ---------------------------------------------------------

def parse_args():
    """
    Parses command-line arguments for model training configuration.

    Returns:
        argparse.Namespace: A namespace containing all parsed configuration arguments.
    """
    parser = argparse.ArgumentParser()

    # Model selection
    parser.add_argument('--name', default="UNET",
                        choices=['UNET', 'NestedUNET'],
                        help='Model architecture to use: UNET or NestedUNET.')

    # Training parameters
    parser.add_argument('--epochs', default=70, type=int, metavar='N',
                        help='Total number of training epochs.')
    parser.add_argument('-b', '--batch_size', default=8, type=int,
                        metavar='N', help='Mini-batch size for training.')

    parser.add_argument('--early_stopping', default=20, type=int,
                        metavar='N', help='Number of epochs to trigger early stopping.')

    parser.add_argument('--num_workers', default=12, type=int,
                        help='Number of data loading worker threads.')

    # Optimizer settings
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='Optimizer to use: Adam or SGD.')
    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float,
                        metavar='LR', help='Initial learning rate.')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum factor for SGD (not used with Adam).')
    parser.add_argument('--weight_decay', default=0.0001, type=float,
                        help='Weight decay (L2 regularization).')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='Use Nesterov momentum with SGD.')

    # Data augmentation
    parser.add_argument('--augmentation', type=str2bool, default=False,
                        choices=[True, False],
                        help='Enable data augmentation.')

    config = parser.parse_args()
    return config



def train(train_loader, model, criterion, optimizer):
    """
    Trains the model for one epoch.

    Args:
        train_loader (DataLoader): Dataloader for training samples.
        model (torch.nn.Module): Model to be trained.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer used to update model weights.

    Returns:
        OrderedDict: Average training loss, IOU score, and Dice coefficient for the epoch.
    """
    # Initialize average meters to track metrics
    avg_meters = {
        'loss': AverageMeter(),
        'iou': AverageMeter(),
        'dice': AverageMeter()
    }

    model.train()  # Set model to training mode

    # Progress bar for training
    pbar = tqdm(total=len(train_loader), desc="Training", disable=(dist.get_rank() != 0))

    # Iterate over training data
    for input, target in train_loader:
        input = input.cuda()
        target = target.cuda()

        # Forward pass
        output = model(input)

        # Compute loss and evaluation metrics
        loss = criterion(output, target)
        iou = iou_score(output, target)
        dice = dice_coef(output, target)

        dist.barrier()  # Ensure synchronization across all distributed processes

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update average metrics
        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))
        avg_meters['dice'].update(dice, input.size(0))

        # Update progress bar
        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
            ('dice', avg_meters['dice'].avg)
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)

    pbar.close()

    return OrderedDict([
        ('loss', avg_meters['loss'].avg),
        ('iou', avg_meters['iou'].avg),
        ('dice', avg_meters['dice'].avg)
    ])


def validate(val_loader, model, criterion):
    """
    Evaluates the model on the validation set.

    Args:
        val_loader (DataLoader): Dataloader for validation samples.
        model (torch.nn.Module): Model to be evaluated.
        criterion (torch.nn.Module): Loss function.

    Returns:
        OrderedDict: Average validation loss, IOU score, and Dice coefficient.
    """
    # Initialize average meters to track metrics
    avg_meters = {
        'loss': AverageMeter(),
        'iou': AverageMeter(),
        'dice': AverageMeter()
    }

    model.eval()  # Set model to evaluation mode

    with torch.no_grad():
        # Progress bar for validation
        pbar = tqdm(total=len(val_loader), desc="Validation", disable=(dist.get_rank() != 0))

        # Iterate over validation data
        for input, target in val_loader:
            input = input.cuda().float()
            target = target.cuda().float()

            # Forward pass
            output = model(input)

            # Compute loss and evaluation metrics
            loss = criterion(output, target)
            iou = iou_score(output, target)
            dice = dice_coef(output, target)

            # Update average metrics
            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            # Update progress bar
            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)

        pbar.close()

    return OrderedDict([
        ('loss', avg_meters['loss'].avg),
        ('iou', avg_meters['iou'].avg),
        ('dice', avg_meters['dice'].avg)
    ])

def save_checkpoint(state, filename):
    """
    Saves the current training state (model and optimizer) to a file.

    Args:
        state (dict): Dictionary containing model and optimizer states.
        filename (str): Path to save the checkpoint.
    """
    if dist.get_rank() == 0:
        torch.save(state, filename)


def load_checkpoint(filename, model, optimizer):
    """
    Loads a saved checkpoint into the model and optimizer.

    Args:
        filename (str): Path to the checkpoint file.
        model (torch.nn.Module): Model to load the weights into.
        optimizer (torch.optim.Optimizer): Optimizer to load the state into.

    Returns:
        tuple: (epoch, best_dice) where epoch is the last completed epoch and best_dice is the best validation dice score achieved.
    """
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  # Scheduler loading is commented out
    return checkpoint['epoch'], checkpoint['best_dice']


def setup_ddp():
    """
    Initializes the Distributed Data Parallel (DDP) environment.

    Returns:
        torch.device: The current device assigned to this process.
    """
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    return device


def cleanup_ddp():
    """
    Cleans up the Distributed Data Parallel (DDP) environment.
    """
    dist.destroy_process_group()


def reduce_mean(tensor):
    """
    Reduces the input tensor across all GPUs by averaging.

    Args:
        tensor (torch.Tensor): Tensor to be averaged across devices.

    Returns:
        torch.Tensor: Averaged tensor.
    """
    if tensor.is_cuda:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return tensor


def main():
    """
    Main training loop for distributed multi-GPU training.
    Handles model setup, checkpointing, K-Fold cross-validation, 
    training, validation, and logging.
    """
    # ----------------- Distributed Setup -----------------
    device = setup_ddp()

    if dist.get_rank() == 0:
        print(f"Distributed training initialized with {dist.get_world_size()} processes.")

    config = vars(parse_args())

    # Print system information (only by rank 0)
    if dist.get_rank() == 0:
        print("CUDA Available:", torch.cuda.is_available())
        print("CUDA Version:", torch.version.cuda)
        print("cuDNN Version:", torch.backends.cudnn.version())
        print("Allocated Memory:", torch.cuda.memory_allocated())
        print("Reserved Memory:", torch.cuda.memory_reserved())

    # ----------------- Output Directory Setup -----------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]

    if config['augmentation']:
        file_name = f"{config['name']}_with_augmentation_{timestamp}_{unique_id}"
    else:
        file_name = f"{config['name']}_base_{timestamp}_{unique_id}"

    if dist.get_rank() == 0:
        os.makedirs(f'model_outputs/{file_name}', exist_ok=True)
        print(f"Creating directory: model_outputs/{file_name}")

        # Save configuration settings to YAML
        config_path = f'model_outputs/{file_name}/config.yml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        print("Config file created.")

    # Ensure all processes wait until directory and config are ready
    dist.barrier()

    if dist.get_rank() == 0:
        print('-' * 20)
        print("Configuration settings:")
        for key, value in config.items():
            print(f'{key}: {value}')
        print('-' * 20)

    # ----------------- Loss Function -----------------
    criterion = BCEDiceFocalLoss( # By default, this is set to BCEDiceFocalLoss
        alpha=1.0,
        gamma=2.0,
        focal_weight=0.5
    ).cuda()

    cudnn.benchmark = True

    # ----------------- Model Setup -----------------
    if dist.get_rank() == 0:
        print("=> Creating model...")

    if config['name'] == 'NestedUNET':
        model = NestedUNet(num_classes=1)
    else:
        model = UNet(n_channels=1, n_classes=1, bilinear=True)
    
    model = model.to(device)

    # Wrap model in DistributedDataParallel (DDP)
    if torch.cuda.device_count() > 1:
        if dist.get_rank() == 0:
            print(f"Using {torch.cuda.device_count()} GPUs.")
        model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])])

    # ----------------- Optimizer and Scheduler -----------------
    params = filter(lambda p: p.requires_grad, model.parameters())

    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError("Unsupported optimizer selected.")

    scheduler = StepLR(optimizer, step_size=40, gamma=0.8)

    # ----------------- Dataset Preparation -----------------
    IMAGE_DIR = '/dcs/22/u2202609/year_3/cs310/Project/Preprocessing/data/Image/' # You should set your own paths here to the correct Preprocessing directory.
    MASK_DIR = '/dcs/22/u2202609/year_3/cs310/Project/Preprocessing/data/Mask/'
    meta = pd.read_csv('/dcs/22/u2202609/year_3/cs310/Project/Preprocessing/csv/meta.csv')

    # Attach absolute file paths
    meta['original_image'] = meta['original_image'].apply(lambda x: IMAGE_DIR + x + '.npy')
    meta['mask_image'] = meta['mask_image'].apply(lambda x: MASK_DIR + x + '.npy')

    all_meta = meta[meta['data_split'].isin(['Train', 'Validation'])].reset_index(drop=True)

    # ----------------- Checkpoint Loading -----------------
    checkpoint_filename = f'model_outputs/{file_name}/checkpoint.pth'
    log_filename = f'model_outputs/{file_name}/log.csv'

    dist.barrier()

    if os.path.exists(checkpoint_filename):
        print(f"=> Loading checkpoint from {checkpoint_filename}")
        start_epoch, best_dice = load_checkpoint(checkpoint_filename, model, optimizer)
        log = pd.read_csv(log_filename)
    else:
        start_epoch = 0
        best_dice = 0
        log = pd.DataFrame(columns=['epoch', 'lr', 'loss', 'iou', 'dice', 'val_loss', 'val_iou', 'val_dice'])

    # ----------------- Prepare K-Fold Cross Validation -----------------
    all_image_paths = list(all_meta['original_image'])
    all_mask_paths = list(all_meta['mask_image'])

    kf = KFold(n_splits=9, shuffle=True, random_state=26)

    torch.cuda.reset_peak_memory_stats(device)
    epoch_start_time = time.time()
    trigger = 0

    # ----------------- Training Loop -----------------
    for idx, epoch in enumerate(range(start_epoch, config['epochs'])):
        if dist.get_rank() == 0:
            print(f"\nEpoch {idx}:")

        # Initialize storage for epoch metrics
        epoch_metrics = {
            'loss': torch.zeros(1).cuda(),
            'iou': torch.zeros(1).cuda(),
            'dice': torch.zeros(1).cuda(),
            'val_loss': torch.zeros(1).cuda(),
            'val_iou': torch.zeros(1).cuda(),
            'val_dice': torch.zeros(1).cuda()
        }
        fold_count = torch.zeros(1).cuda()

        # ----------------- Fold Loop -----------------
        for fold, (train_idx, val_idx) in enumerate(kf.split(all_image_paths)):
            if fold % dist.get_world_size() != dist.get_rank():
                continue

            if dist.get_rank() == 0:
                print(f"Starting Fold {fold + 1} of {kf.get_n_splits()}")

            # Prepare train/validation datasets
            train_image_paths = [all_image_paths[i] for i in train_idx]
            train_mask_paths = [all_mask_paths[i] for i in train_idx]
            val_image_paths = [all_image_paths[i] for i in val_idx]
            val_mask_paths = [all_mask_paths[i] for i in val_idx]

            train_dataset = MyLidcDataset(train_image_paths, train_mask_paths, config['augmentation'])
            val_dataset = MyLidcDataset(val_image_paths, val_mask_paths, config['augmentation'])

            # Distributed samplers
            train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
            val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)

            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=config['batch_size'], sampler=train_sampler,
                pin_memory=True, drop_last=True, num_workers=config['num_workers']
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=config['batch_size'], sampler=val_sampler,
                pin_memory=True, drop_last=False, num_workers=config['num_workers']
            )

            # Train and validate
            train_log = train(train_loader, model, criterion, optimizer)
            for key in train_log:
                train_log[key] = reduce_mean(torch.tensor(train_log[key]).cuda()).item()

            val_log = validate(val_loader, model, criterion)
            for key in val_log:
                val_log[key] = reduce_mean(torch.tensor(val_log[key]).cuda()).item()

            # Accumulate fold metrics
            for key in epoch_metrics:
                epoch_metrics[key] += torch.tensor(train_log.get(key, val_log.get(key, 0))).cuda()

            fold_count += 1
            print(f"Rank {dist.get_rank()} finished Fold {fold + 1}")

            dist.barrier()

        # Synchronize metrics across all devices
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1024**2

        for key in epoch_metrics:
            dist.all_reduce(epoch_metrics[key], op=dist.ReduceOp.SUM)
        dist.all_reduce(fold_count, op=dist.ReduceOp.SUM)

        avg_metrics = {key: (epoch_metrics[key] / fold_count).item() for key in epoch_metrics.keys()}

        if dist.get_rank() == 0:
            print(f'Epoch [{epoch + 1}/{config["epochs"]}] - Avg Training Loss: {avg_metrics["loss"]:.4f}, '
                  f'Avg Dice: {avg_metrics["dice"]:.4f}, Avg IOU: {avg_metrics["iou"]:.4f}')

        # ----------------- Logging -----------------
        if dist.get_rank() == 0:
            if os.path.exists(log_filename):
                log = pd.read_csv(log_filename)
                last_logged_epoch = log['epoch'].max()
            else:
                log = pd.DataFrame(columns=[
                    'epoch', 'lr', 'loss', 'iou', 'dice',
                    'val_loss', 'val_iou', 'val_dice',
                    'epoch_time_s', 'peak_mem_MB'
                ])
                last_logged_epoch = 0

            if epoch + 1 > last_logged_epoch:
                new_row = {
                    'epoch': epoch + 1,
                    'lr': config['lr'],
                    'loss': avg_metrics['loss'],
                    'iou': avg_metrics['iou'],
                    'dice': avg_metrics['dice'],
                    'val_loss': avg_metrics['val_loss'],
                    'val_iou': avg_metrics['val_iou'],
                    'val_dice': avg_metrics['val_dice'],
                    'epoch_time_s': epoch_duration,
                    'peak_mem_MB': peak_mem_mb
                }
                log = pd.concat([log, pd.DataFrame([new_row])], ignore_index=True)
                log.to_csv(log_filename, index=False)
                print(f"Epoch {epoch + 1} logged successfully.")

        dist.barrier()

        # ----------------- Model Checkpointing -----------------
        if avg_metrics['val_dice'] > best_dice:
            if dist.get_rank() == 0:
                model_filename = f'model_outputs/{file_name}/model.pth'
                torch.save(model.state_dict(), model_filename)
                print("=> Best model saved (Dice improved)")
            best_dice = avg_metrics['val_dice']
            trigger = 0
        else:
            trigger += 1

        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_dice': best_dice
        }, checkpoint_filename)

        # ----------------- Early Stopping -----------------
        if trigger >= config['early_stopping']:
            if dist.get_rank() == 0:
                print("=> Early stopping triggered")
            break

        scheduler.step()
        torch.cuda.empty_cache()

    cleanup_ddp()


if __name__ == '__main__':
    main()
