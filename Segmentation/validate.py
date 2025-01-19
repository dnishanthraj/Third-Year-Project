import pandas as pd
import argparse
import os
from glob import glob
from collections import OrderedDict
import numpy as np

import matplotlib.pyplot as plt
import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import yaml
from sklearn.model_selection import train_test_split
from scipy import ndimage as ndi
from scipy.ndimage import label, generate_binary_structure
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm
from grad_cam import GradCAM


from dataset import MyLidcDataset
from metrics import iou_score,dice_coef,dice_coef2
from utils import AverageMeter, str2bool

from Unet.unet_model import UNet
from UnetNested.Nested_Unet import NestedUNet

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default="UNET",
                        help='model name: UNET',choices=['UNET', 'NestedUNET'])
    # Get augmented version?
    parser.add_argument('--augmentation',default=False,type=str2bool,
                help='Shoud we get the augmented version?')
    parser.add_argument('--folder', required=True,
                help='Path to the folder containing model and config files')

    args = parser.parse_args()

    return args

# NOTE: A green overlay for the predicted mask means that we have no predicted mask (for some unknown reason)
def save_output(output, output_directory, test_image_paths, counter):
    label = test_image_paths[counter][-23:]
    label = label.replace('NI', 'PD').replace('.npy', '.png')

    # Load the original image
    original_image_path = test_image_paths[counter]
    original_image = np.load(original_image_path)
    original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())  # Normalize

    # Overlay the predicted mask on the original image
    plt.figure(figsize=(10, 10))
    plt.imshow(original_image, cmap="gray")  # Original image in grayscale
    plt.imshow(output[0, :, :], cmap="jet", alpha=0.5)  # Process a single image
    plt.colorbar()
    plt.title("Predicted Mask Overlay")

    # Save the overlay with the same naming convention
    overlay_save_path = os.path.join(output_directory, label)
    os.makedirs(output_directory, exist_ok=True)
    plt.savefig(overlay_save_path)
    plt.close()


def save_grad_cam(output, grad_cam_dir, test_image_paths, counter, grad_cam_generator):
    grad_cam_label = test_image_paths[counter][-23:]
    grad_cam_label = grad_cam_label.replace('NI', 'GC').replace('.npy', '.png')

    # Load the original image
    original_image_path = test_image_paths[counter]
    original_image = np.load(original_image_path)
    original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())  # Normalize

    # Enable gradient computation for Grad-CAM
    with torch.set_grad_enabled(True):
        heatmap = grad_cam_generator.generate(
            torch.tensor(output[0, :, :]).unsqueeze(0).unsqueeze(0).cuda(),
            class_idx=0
        )

    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    # Overlay Grad-CAM heatmap on the original image
    plt.figure(figsize=(10, 10))
    plt.imshow(original_image, cmap="gray")
    plt.imshow(heatmap, cmap="jet", alpha=0.5)
    plt.colorbar()
    plt.title("Grad-CAM Heatmap Overlay")

    grad_cam_save_path = os.path.join(grad_cam_dir, grad_cam_label)
    os.makedirs(grad_cam_dir, exist_ok=True)
    plt.savefig(grad_cam_save_path)
    plt.close()


def calculate_fp(prediction_dir,mask_dir,distance_threshold=80):
    """This calculates the fp by comparing the predicted mask and orginal mask"""
    #TP,TN,FP,FN
    #FN will always be zero here as all the mask contains a nodule
    confusion_matrix =[0,0,0,0]
    # This binary structure enables the function to recognize diagnoally connected label as same nodule.
    s = generate_binary_structure(2,2)
    print('Length of prediction dir is ',len(os.listdir(prediction_dir)))
    for prediction in os.listdir(prediction_dir):
        #print(confusion_matrix)
        pid = 'LIDC-IDRI-'+prediction[:4]
        mask_id = prediction.replace('PD','MA')
        mask = np.load(mask_dir+'/'+pid+'/'+mask_id)
        predict = np.load(prediction_dir+'/'+prediction)
        answer_com = np.array(ndi.center_of_mass(mask))
        # Patience is used to check if the patch has cropped the same image
        patience =0
        labeled_array, nf = label(predict, structure=s)
        if nf>0:
            for n in range(nf):
                lab=np.array(labeled_array)
                lab[lab!=(n+1)]=0
                lab[lab==(n+1)]=1
                predict_com=np.array(ndi.center_of_mass(labeled_array))
                if np.linalg.norm(predict_com-answer_com,2) < distance_threshold:
                    patience +=1
                else:
                    confusion_matrix[2]+=1
            if patience > 0:
                # Add to True Positive
                confusion_matrix[0]+=1
            else:
                # Add to False Negative
                # if the patience remains 0, and nf >0, it means that the slice contains both the TN and FP
                confusion_matrix[3]+=1

        else:
            # Add False Negative since the UNET didn't detect a cancer even when there was one
            confusion_matrix[3]+=1
    return np.array(confusion_matrix)

def calculate_fp_clean_dataset(prediction_dir,distance_threshold=80):
    """This calculates the confusion matrix for clean dataset"""
    #TP,TN,FP,FN
    #When we calculate the confusion matrix for clean dataset, we can only get TP and FP.
    # TP - There is no nodule, and the segmentation model predicted there is no nodule
    # FP - There is no nodule, but the segmentation model predicted there is a nodule
    confusion_matrix =[0,0,0,0]
    s = generate_binary_structure(2,2)
    for prediction in os.listdir(prediction_dir):
        predict = np.load(prediction_dir+'/'+prediction)
        # Patience is used to check if the patch has cropped the same image
        patience =0
        labeled_array, nf = label(predict, structure=s)
        if nf>0:
            previous_com = np.array([-1,-1])
            for n in range(nf):
                lab=np.array(labeled_array)
                lab[lab!=(n+1)]=0
                lab[lab==(n+1)]=1
                predict_com=np.array(ndi.center_of_mass(labeled_array))
                if previous_com[0] == -1:
                    # add to false positive
                    confusion_matrix[2]+=1
                    previous_com = predict_com
                    continue
                else:
                    if np.linalg.norm(previous_com-predict_com,2) > distance_threshold:
                        if patience != 0:
                            #print("This nodule has already been taken into account")
                            continue
                        # add false positive
                        confusion_matrix[2]+=1
                        patience +=1

        else:
            # Add True Negative since the UNET didn't detect a cancer even when there was one
            confusion_matrix[1]+=1

    return np.array(confusion_matrix)

def main():
    args = vars(parse_args())

    if args['augmentation']== True:
        NAME = args['name'] + '_with_augmentation'
    else:
        NAME = args['name'] +'_base'

    # load configuration

    folder = args['folder']  # Folder path passed as argument
    # config_path = os.path.join(folder, 'config.yml')
    # model_path = os.path.join(folder, 'model.pth')
    # config_path = os.path.join('/model_outputs', folder, 'config.yml')
    # model_path = os.path.join('/model_outputs', folder, 'model.pth')

    config_path = os.path.join(os.getcwd(), 'model_outputs', folder, 'config.yml')
    model_path = os.path.join(os.getcwd(), 'model_outputs', folder, 'model.pth')

    with open(config_path, 'r') as f:
        # config = yaml.load(f)
        config = yaml.load(f, Loader=yaml.SafeLoader)


    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)


    cudnn.benchmark = True

    # create model
    print("=> creating model {}".format(NAME))
    if config['name']=='NestedUNET':
        model = NestedUNet(num_classes=1)
    else:
        model = UNet(n_channels=1, n_classes=1, bilinear=True)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    print("Loading model file from {}".format(NAME))
    

    
    # state_dict = torch.load('model_outputs/{}/model.pth'.format(NAME), weights_only=True)
    state_dict = torch.load(model_path, weights_only=True, map_location=torch.device('cuda'))

    # Strip `module.` prefix if present in keys
    #Fixed to use multiple GPU's

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.') and not any(name.startswith('module.') for name in model.state_dict().keys()):
            # Strip 'module.' if not expected by the model
            new_state_dict[k[7:]] = v
        elif not k.startswith('module.') and any(name.startswith('module.') for name in model.state_dict().keys()):
            # Add 'module.' if expected by the model
            new_state_dict[f'module.{k}'] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)

    # model.load_state_dict(torch.load('model_outputs/{}/model.pth'.format(NAME)))
    model = model.cuda()

    # Use the deepest encoder layer for Grad-CAM
    grad_cam = GradCAM(model, model.conv2_0)  # Adjust `model.conv2_0` to the appropriate layer


    # Data loading code
    IMAGE_DIR = '/dcs/22/u2202609/year_3/cs310/Project/Preprocessing/data/Image/'
    MASK_DIR = '/dcs/22/u2202609/year_3/cs310/Project/Preprocessing/data/Mask/'
    #Meta Information
    meta = pd.read_csv('/dcs/22/u2202609/year_3/cs310/Project/Preprocessing/csv/meta.csv')
    # Get train/test label from meta.csv
    meta['original_image']= meta['original_image'].apply(lambda x:IMAGE_DIR+ x +'.npy')
    meta['mask_image'] = meta['mask_image'].apply(lambda x:MASK_DIR+ x +'.npy')
    test_meta = meta[meta['data_split']=='Test']

    # Get all *npy images into list for Test(True Positive Set)
    test_image_paths = list(test_meta['original_image'])
    test_mask_paths = list(test_meta['mask_image'])

    total_patients = len(test_meta.groupby('patient_id'))

    print("*"*50)
    print("The lenght of image: {}, mask folders: {} for test".format(len(test_image_paths),len(test_mask_paths)))
    print("Total patient number is :{}".format(total_patients))


    # Directory to save U-Net predict output
    # OUTPUT_MASK_DIR = '/home/LUNG_DATA/Segmentation_output/{}'.format(NAME) #Changed this
    # OUTPUT_MASK_DIR = '/dcs/22/u2202609/year_3/cs310/Project/Segmentation/Segmentation_output/{}'.format(NAME)
    OUTPUT_MASK_DIR = os.path.join(os.getcwd(), 'model_outputs', folder, 'Segmentation_output', NAME)
    GRAD_CAM_DIR = os.path.join(os.getcwd(), 'model_outputs', folder, 'Grad_CAM_output', NAME)
    print("Saving OUTPUT files in directory {}".format(OUTPUT_MASK_DIR))
    print(f"Saving Grad-CAM heatmaps in {GRAD_CAM_DIR}")
    os.makedirs(OUTPUT_MASK_DIR,exist_ok=True)
    os.makedirs(GRAD_CAM_DIR, exist_ok=True)



    test_dataset = MyLidcDataset(test_image_paths, test_mask_paths)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=6)
    model.eval()
    print(" ")
    print("Printing the first 5 image directories...",test_image_paths[:5])
    print("Printing the first 5 mask directories...",test_mask_paths[:5])
    ##########################
    ## Load Clean related ####
    ##########################
    CLEAN_DIR_IMG ='/dcs/22/u2202609/year_3/cs310/Project/Preprocessing/data/Clean/Image/'
    CLEAN_DIR_MASK ='/dcs/22/u2202609/year_3/cs310/Project/Preprocessing/data/Clean/Mask/'
    clean_meta = pd.read_csv('/dcs/22/u2202609/year_3/cs310/Project/Preprocessing/csv/clean_meta.csv')
    # Get train/test label from clean_meta.csv
    clean_meta['original_image']= clean_meta['original_image'].apply(lambda x:CLEAN_DIR_IMG+ x +'.npy')
    clean_meta['mask_image'] = clean_meta['mask_image'].apply(lambda x:CLEAN_DIR_MASK+ x +'.npy')
    clean_test_meta = clean_meta[clean_meta['data_split']=='Test']
    # Get all *npy images into list for Test(True Negative Set)
    clean_test_image_paths = list(clean_test_meta['original_image'])
    clean_test_mask_paths = list(clean_test_meta['mask_image'])

    clean_total_patients = len(clean_test_meta.groupby('patient_id'))
    print("*"*50)
    print("The lenght of clean image: {}, mask folders: {} for clean test set".format(len(clean_test_image_paths),len(clean_test_mask_paths)))
    print("Total patient number is :{}".format(clean_total_patients))
    # Directory to save U-Net predict output for clean dataset


    CLEAN_NAME = 'CLEAN_'+NAME

    # CLEAN_OUTPUT_MASK_DIR = '/home/LUNG_DATA/Segmentation_output/{}'.format(CLEAN_NAME)
    # CLEAN_OUTPUT_MASK_DIR = '/dcs/22/u2202609/year_3/cs310/Project/Segmentation/Segmentation_output/{}'.format(CLEAN_NAME)
    CLEAN_OUTPUT_MASK_DIR = os.path.join(os.getcwd(), 'model_outputs', folder, 'Segmentation_output', CLEAN_NAME)
    CLEAN_GRAD_CAM_DIR = os.path.join(os.getcwd(), 'model_outputs', folder, 'Grad_CAM_output', CLEAN_NAME)
    print("Saving CLEAN files in directory {}".format(CLEAN_OUTPUT_MASK_DIR))
    print(f"Saving CLEAN Grad-CAM heatmaps in {CLEAN_GRAD_CAM_DIR}")
    os.makedirs(CLEAN_GRAD_CAM_DIR, exist_ok=True)
    os.makedirs(CLEAN_OUTPUT_MASK_DIR,exist_ok=True)
    clean_test_dataset = MyLidcDataset(clean_test_image_paths, clean_test_mask_paths)
    clean_test_loader = torch.utils.data.DataLoader(
        clean_test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=6)

    avg_meters = {'iou': AverageMeter(),
                  'dice': AverageMeter()}


    with torch.no_grad():
        counter = 0  # Initialize counter
        pbar = tqdm(total=len(test_loader))
        
        for input, target in test_loader:
            input = input.cuda()
            target = target.cuda()

            output = model(input)
            iou = iou_score(output, target)
            dice = dice_coef2(output, target)

            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg)
            ])

            output = torch.sigmoid(output)
            output = (output > 0.5).float().cpu().numpy() # Thresholding - changed to 0.3 to see if better accuracy
            output = np.squeeze(output, axis=1)

            # Process one image at a time, ensuring `counter` is consistent
            for i in range(output.shape[0]):
                # Save predicted mask
                save_output(output[i:i+1], OUTPUT_MASK_DIR, test_image_paths, counter)
                # Save Grad-CAM heatmap
                save_grad_cam(output[i:i+1], GRAD_CAM_DIR, test_image_paths, counter, grad_cam)
                counter += 1  # Increment only after both operations are complete

            pbar.set_postfix(postfix)
            pbar.update(1)

        pbar.close()



    print("="*50)
    print('IoU: {:.4f}'.format(avg_meters['iou'].avg))
    print('DICE:{:.4f}'.format(avg_meters['dice'].avg))


    confusion_matrix = calculate_fp(OUTPUT_MASK_DIR ,MASK_DIR,distance_threshold=80)
    print("="*50)
    print("TP: {} FP:{}".format(confusion_matrix[0],confusion_matrix[2]))
    print("FN: {} TN:{}".format(confusion_matrix[3],confusion_matrix[1]))
    print("{:2f} FP/per Scan ".format(confusion_matrix[2]/total_patients))
    print("="*50)
    print(" ")
    print("NOW, INCLUDE CLEAN TEST SET")
    with torch.no_grad():

        counter = 0
        pbar = tqdm(total=len(clean_test_loader))
        for input, target in clean_test_loader:
            input = input.cuda()
            target = target.cuda()

            output = model(input)
            iou = iou_score(output, target)
            dice = dice_coef2(output, target)

            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('iou', avg_meters['iou'].avg),
                ('dice',avg_meters['dice'].avg)
            ])

            output = torch.sigmoid(output)
            output = (output>0.5).float().cpu().numpy()
            output = np.squeeze(output,axis=1)
            #print(output.shape)


            for i in range(output.shape[0]):
                # Save predicted mask
                save_output(output[i:i+1], CLEAN_OUTPUT_MASK_DIR, clean_test_image_paths, counter)
                # Save Grad-CAM heatmap
                save_grad_cam(output[i:i+1], CLEAN_GRAD_CAM_DIR, clean_test_image_paths, counter, grad_cam)
                counter += 1  # Increment only after both operations are complete

            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()
    print("="*50)
    print('IoU: {:.4f}'.format(avg_meters['iou'].avg))
    print('DICE:{:.4f}'.format(avg_meters['dice'].avg))
    clean_confusion_matrix = calculate_fp_clean_dataset(CLEAN_OUTPUT_MASK_DIR)
    print(clean_confusion_matrix)
    confusion_matrix_total = clean_confusion_matrix + confusion_matrix
    total_patients += clean_total_patients
    print("="*50)
    print("TP: {} FP:{}".format(confusion_matrix_total[0],confusion_matrix_total[2]))
    print("FN: {} TN:{}".format(confusion_matrix_total[3],confusion_matrix_total[1]))
    print("{:2f} FP/per Scan ".format(confusion_matrix_total[2]/total_patients))
    print("Number of total patients used for test are {}, among them clean patients are {}".format(total_patients,clean_total_patients))
    print("="*50)
    print(" ")



    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()