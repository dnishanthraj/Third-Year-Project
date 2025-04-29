#!/usr/bin/env python

# ----------------------------
# Imports
# ----------------------------
import argparse
import os
from glob import glob
from collections import OrderedDict
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import yaml
import time
import joblib
import xgboost as xgb
from tqdm import tqdm

# Morphological analysis utilities
from skimage.measure import regionprops, label as sklabel
from scipy.ndimage import center_of_mass, generate_binary_structure, label as ndi_label

# Model architectures
from Unet.unet_model import UNet
from UnetNested.Nested_Unet import NestedUNet  # Note: naming matches usage below

# Local modules
from dataset import MyLidcDataset
from metrics import (
    iou_score, dice_coef2, calculate_fp, calculate_fp_clean_dataset, 
    calculate_accuracy, calculate_f1_score, calculate_specificity, 
    calculate_precision, calculate_recall, calculate_fpps, save_metrics_to_csv
)
from utils import AverageMeter, str2bool
from grad_cam import GradCAM

# Cross-validation
from sklearn.model_selection import KFold, cross_validate

# ----------------------------
# Argument Parser
# ----------------------------
def parse_args():
    """
    Parses command-line arguments for validation script.

    Returns:
        argparse.Namespace: Parsed arguments including model name, 
                            augmentation flag, output folder, 
                            and distance threshold for FPR classification.
    """
    parser = argparse.ArgumentParser()

    # Model configuration
    parser.add_argument('--name', default="UNET", choices=['UNET', 'NestedUNET'],
                        help='Model name to evaluate (UNET or NestedUNET)')
    parser.add_argument('--augmentation', default=False, type=str2bool,
                        help='Whether augmentation was used during training')

    # Folder paths
    parser.add_argument('--folder', required=True,
                        help='Folder containing the trained model and config files')

    # False Positive Reduction threshold
    parser.add_argument('--distance_threshold', type=float, default=80,
                        help='Distance threshold (in pixels) for FPR classifier')

    return parser.parse_args()


#############################
# Helper Functions
#############################

def is_true_detection(pred_mask, gt_mask, distance_threshold=80):
    """
    Determines whether a predicted connected component is a true detection.

    Args:
        pred_mask (np.ndarray): Predicted binary mask.
        gt_mask (np.ndarray): Ground truth binary mask.
        distance_threshold (float): Maximum allowed distance between 
                                    predicted and ground truth centers of mass.

    Returns:
        int: 1 if a predicted region is within the threshold of the ground truth center, else 0.
    """
    if np.sum(gt_mask) == 0:
        return 0

    structure = generate_binary_structure(2, 2)
    gt_com = np.array(center_of_mass(gt_mask))
    labeled_array, _ = ndi_label(pred_mask, structure=structure)

    if np.max(labeled_array) == 0:
        return 0

    num_features = np.max(labeled_array)
    for n in range(num_features):
        region = (labeled_array == (n + 1)).astype(np.uint8)
        pred_com = np.array(center_of_mass(region))
        if np.linalg.norm(pred_com - gt_com, 2) < distance_threshold:
            return 1

    return 0


def extract_morphological_features(mask):
    """
    Extracts morphological features from a binary mask.

    Features include: area, perimeter, eccentricity, solidity, and compactness.

    Args:
        mask (np.ndarray): Binary mask.

    Returns:
        list: [area, perimeter, eccentricity, solidity, compactness].
    """
    labeled_mask, _ = ndi_label(mask)
    props = regionprops(labeled_mask)

    if len(props) == 0:
        return [0, 0, 0, 0, 0]

    largest = max(props, key=lambda p: p.area)
    area = largest.area
    perimeter = largest.perimeter if largest.perimeter > 0 else 1
    eccentricity = largest.eccentricity
    solidity = largest.solidity
    compactness = area / (perimeter ** 2)

    return [area, perimeter, eccentricity, solidity, compactness]


def save_output(output, output_directory, test_image_paths, counter):
    """
    Saves predicted output masks to disk.

    Args:
        output (np.ndarray): Predicted mask (single sample).
        output_directory (str): Directory where outputs will be saved.
        test_image_paths (list): List of original test image paths.
        counter (int): Current index in test set.
    """
    file_label = test_image_paths[counter][-23:].replace('NI', 'PD').replace('.npy', '.npy')
    save_path = os.path.join(output_directory, file_label)

    os.makedirs(output_directory, exist_ok=True)
    np.save(save_path, output[0, :, :])


def save_grad_cam(output, grad_cam_dir, test_image_paths, counter, grad_cam_generator):
    """
    Saves Grad-CAM heatmaps for interpretability.

    Args:
        output (np.ndarray): Predicted mask (single sample).
        grad_cam_dir (str): Directory where Grad-CAM outputs will be saved.
        test_image_paths (list): List of original test image paths.
        counter (int): Current index in test set.
        grad_cam_generator (GradCAM): GradCAM instance for generating heatmaps.
    """
    file_label = test_image_paths[counter][-23:].replace('NI', 'GC').replace('.npy', '.npy')

    with torch.set_grad_enabled(True):
        heatmap = grad_cam_generator.generate(
            torch.tensor(output[0, :, :]).unsqueeze(0).unsqueeze(0).cuda(),
            class_idx=0
        )

    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())  # Normalize heatmap
    grad_cam_save_path = os.path.join(grad_cam_dir, file_label)

    os.makedirs(grad_cam_dir, exist_ok=True)
    np.save(grad_cam_save_path, heatmap)


def apply_fpr_classifier(pred_mask, classifier, distance_threshold=80):
    """
    For each connected candidate in pred_mask, extracts morphological features and forms a feature vector. Uses the classifier to predict a label.
    If predicted label is 0 (false positive), removes that candidate from the mask.
    Returns the modified mask.
    """
    structure = generate_binary_structure(2, 2)
    labeled_array, _ = ndi_label(pred_mask, structure=structure)
    if np.max(labeled_array) == 0:
        return pred_mask
    modified_mask = np.copy(pred_mask)
    num_features = np.max(labeled_array)
    for n in range(num_features):
        candidate = (labeled_array == (n + 1)).astype(np.uint8)
        morph_feats = extract_morphological_features(candidate)
        feature_vector = np.array(morph_feats).reshape(1, -1)
        pred_label = classifier.predict(feature_vector)
        if pred_label[0] == 0:
            modified_mask[candidate == 1] = 0
    return modified_mask


#############################
# FP Classifier Training 
#############################
def train_fpr_classifier(fp_out_dir, meta_csv, clean_meta_csv, IMAGE_DIR, MASK_DIR, CLEAN_DIR_IMG, CLEAN_DIR_MASK, model):
    print("Training FP classifier (no pre-trained classifier found)...")
    meta = pd.read_csv(meta_csv)
    clean_meta = pd.read_csv(clean_meta_csv)
    meta['original_image'] = meta['original_image'].apply(lambda x: IMAGE_DIR + x + '.npy')
    meta['mask_image'] = meta['mask_image'].apply(lambda x: MASK_DIR + x + '.npy')
    clean_meta['original_image'] = clean_meta['original_image'].apply(lambda x: CLEAN_DIR_IMG + x + '.npy')
    clean_meta['mask_image'] = clean_meta['mask_image'].apply(lambda x: CLEAN_DIR_MASK + x + '.npy')
    
    # Use both Train and Validation as training set
    normal_train = meta[meta['data_split'].isin(['Train', 'Validation'])].copy()
    clean_train = clean_meta[clean_meta['data_split'].isin(['Train', 'Validation'])].copy()
    
    features_list = []
    print("Extracting features from normal training samples...")
    for idx, row in tqdm(normal_train.iterrows(), total=len(normal_train), desc="Normal features"):
        try:
            image = np.load(row['original_image'])
            gt_mask = np.load(row['mask_image'])
        except Exception as e:
            print(f"Error loading {row['original_image']} or {row['mask_image']}: {e}")
            continue
        image_tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0).float().cuda()
        with torch.no_grad():
            output = model(image_tensor)
            output = torch.sigmoid(output)
            pred_mask = (output > 0.5).float().cpu().numpy()[0, 0]
        detection_label = is_true_detection(pred_mask, gt_mask, distance_threshold=80)
        morph_feats = extract_morphological_features(pred_mask)
        
        feat_dict = {
            'area': morph_feats[0],
            'perimeter': morph_feats[1],
            'eccentricity': morph_feats[2],
            'solidity': morph_feats[3],
            'compactness': morph_feats[4],
            'label': detection_label
        }
        features_list.append(feat_dict)
        
    print("Extracting features from clean training samples...")
    for idx, row in tqdm(clean_train.iterrows(), total=len(clean_train), desc="Clean features"):
        try:
            image = np.load(row['original_image'])
            gt_mask = np.load(row['mask_image'])
        except Exception as e:
            print(f"Error loading {row['original_image']} or {row['mask_image']}: {e}")
            continue
        image_tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0).float().cuda()
        with torch.no_grad():
            output = model(image_tensor)
            output = torch.sigmoid(output)
            pred_mask = (output > 0.5).float().cpu().numpy()[0, 0]
        if np.sum(pred_mask) == 0:
            continue
        else:
            detection_label = 0  # All candidates in clean images are false positives.
        morph_feats = extract_morphological_features(pred_mask)
        feat_dict = {
            'area': morph_feats[0],
            'perimeter': morph_feats[1],
            'eccentricity': morph_feats[2],
            'solidity': morph_feats[3],
            'compactness': morph_feats[4],
            'label': detection_label
        }
        features_list.append(feat_dict)
    
    features_df = pd.DataFrame(features_list)
    features_csv = os.path.join(fp_out_dir, "features.csv")
    features_df.to_csv(features_csv, index=False)
    print(f"Features saved to {features_csv}")
    
    feature_columns = ['area', 'perimeter', 'eccentricity', 'solidity', 'compactness']
    X = features_df[feature_columns].values
    y = features_df['label'].values

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        objective='binary:logistic',
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    cv_results = cross_validate(clf, X, y, cv=kf,
                                scoring=['precision', 'recall', 'f1', 'accuracy'],
                                return_train_score=True)
    metrics = {
        'Precision': np.mean(cv_results['test_precision']),
        'Recall': np.mean(cv_results['test_recall']),
        'F1': np.mean(cv_results['test_f1']),
        'Accuracy': np.mean(cv_results['test_accuracy'])
    }
    print("Cross-validation metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    clf.fit(X, y)
    model_save_path = os.path.join(fp_out_dir, "xgb_fpr_model.pkl")
    joblib.dump(clf, model_save_path)
    print(f"XGBoost classifier saved as {model_save_path}")
    metrics_save_path = os.path.join(fp_out_dir, "fpr_metrics.csv")
    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Result'])
    metrics_df.to_csv(metrics_save_path, index=False)
    print(f"FPR metrics saved to {metrics_save_path}")
    return clf



#############################
# Main Function
#############################

def main():
    args = vars(parse_args())
    NAME = args['name'] + ('_with_augmentation' if args['augmentation'] else '_base')
    folder = args['folder']
    base_dir = os.getcwd()
    config_path = os.path.join(base_dir, 'model_outputs', folder, 'config.yml')
    model_path = os.path.join(base_dir, 'model_outputs', folder, 'model.pth')

    # Define output subfolders
    fp_out_dir = os.path.join(base_dir, 'model_outputs', folder, 'fp_classifier')
    os.makedirs(fp_out_dir, exist_ok=True)
    OUTPUT_MASK_DIR = os.path.join(base_dir, 'model_outputs', folder, 'Segmentation_output', NAME)
    GRAD_CAM_DIR = os.path.join(base_dir, 'model_outputs', folder, 'Grad_CAM_output', NAME)
    os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)
    os.makedirs(GRAD_CAM_DIR, exist_ok=True)

    METRICS_DIR = os.path.join(base_dir, 'model_outputs', folder, 'metrics')
    os.makedirs(METRICS_DIR, exist_ok=True)

    # Load configuration and segmentation model
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    if args['name'] == 'NestedUNET':
        model = NestedUNet(num_classes=1)
    else:
        model = UNet(n_channels=1, n_classes=1, bilinear=True)
    state_dict = torch.load(model_path, weights_only=True, map_location=torch.device('cuda'))
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.') and not any(name.startswith('module.') for name in model.state_dict().keys()):
            new_state_dict[k[7:]] = v
        elif not k.startswith('module.') and any(name.startswith('module.') for name in model.state_dict().keys()):
            new_state_dict[f'module.{k}'] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model = model.cuda()
    model.eval()

    if args['name'] == 'NestedUNET':
        grad_cam = GradCAM(model, model.conv2_0)  # Nested UNet architecture
    else:
        grad_cam = GradCAM(model, model.down2)  # Normal UNet architecture


    # Fixed directories for data - define your own paths here to the preprocessing directory. 
    IMAGE_DIR = '/dcs/22/u2202609/year_3/cs310/Project/Preprocessing/data/Image/'
    MASK_DIR = '/dcs/22/u2202609/year_3/cs310/Project/Preprocessing/data/Mask/'
    CLEAN_DIR_IMG = '/dcs/22/u2202609/year_3/cs310/Project/Preprocessing/data/Clean/Image/'
    CLEAN_DIR_MASK = '/dcs/22/u2202609/year_3/cs310/Project/Preprocessing/data/Clean/Mask/'

    # Load meta CSVs for test sets (normal and clean)
    meta_df = pd.read_csv('/dcs/22/u2202609/year_3/cs310/Project/Preprocessing/csv/meta.csv')
    meta_df['original_image'] = meta_df['original_image'].apply(lambda x: IMAGE_DIR + x + '.npy')
    meta_df['mask_image'] = meta_df['mask_image'].apply(lambda x: MASK_DIR + x + '.npy')
    test_meta = meta_df[meta_df['data_split'] == 'Test']

    clean_meta_df = pd.read_csv('/dcs/22/u2202609/year_3/cs310/Project/Preprocessing/csv/clean_meta.csv')
    clean_meta_df['original_image'] = clean_meta_df['original_image'].apply(lambda x: CLEAN_DIR_IMG + x + '.npy')
    clean_meta_df['mask_image'] = clean_meta_df['mask_image'].apply(lambda x: CLEAN_DIR_MASK + x + '.npy')
    clean_test_meta = clean_meta_df[clean_meta_df['data_split'] == 'Test']



    # ----------------------------------------
    # Prepare test datasets and DataLoaders
    # ----------------------------------------

    # Extract image and mask paths from normal test set
    test_image_paths = list(test_meta['original_image'])
    test_mask_paths = list(test_meta['mask_image'])
    total_patients = len(test_meta.groupby('patient_id'))  # Total number of patients (grouped by ID)

    # Extract image and mask paths from clean test set
    clean_test_image_paths = list(clean_test_meta['original_image'])
    clean_test_mask_paths = list(clean_test_meta['mask_image'])
    clean_total_patients = len(clean_test_meta.groupby('patient_id'))

    # Initialize normal test dataset
    test_dataset = MyLidcDataset(test_image_paths, test_mask_paths, return_pid=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=12
    )

    # Initialize clean test dataset
    clean_test_dataset = MyLidcDataset(clean_test_image_paths, clean_test_mask_paths, return_pid=True)
    clean_test_loader = torch.utils.data.DataLoader(
        clean_test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=12
    )

    # ----------------------------------------
    # Load or train FP (False Positive) classifier
    # ----------------------------------------

    # Path to trained classifier
    classifier_path = os.path.join(fp_out_dir, "xgb_fpr_model.pkl")

    if not os.path.exists(classifier_path):
        # If classifier does not exist, train a new FPR classifier
        meta_csv = os.path.join(base_dir, '..', 'Preprocessing', 'csv', 'meta.csv')
        clean_meta_csv = os.path.join(base_dir, '..', 'Preprocessing', 'csv', 'clean_meta.csv')

        # Train the classifier using both normal and clean datasets
        classifier = train_fpr_classifier(
            fp_out_dir,
            meta_csv,
            clean_meta_csv,
            IMAGE_DIR,
            MASK_DIR,
            CLEAN_DIR_IMG,
            CLEAN_DIR_MASK,
            model
        )
    else:
        # Otherwise, load existing classifier
        classifier = joblib.load(classifier_path)

    
        #############################
    # First pass: Raw predictions for normal test set
    #############################

    # Initialize average meters for IoU and Dice metrics
    avg_meters = {'iou': AverageMeter(), 'dice': AverageMeter()}
    
    # Initialize dictionary to collect per-patient slice metrics
    per_patient_metrics_raw = {}
    
    # Track total inference time and total number of slices
    total_inference_time = 0.0
    total_slices = 0

    with torch.no_grad():
        counter = 0  # Counter to track number of slices processed
        pbar = tqdm(total=len(test_loader), desc="Raw predictions (Normal)")  # Progress bar
        raw_predictions = []

        for (input, target, pids) in test_loader:
            input = input.cuda()    # Move input images to GPU
            target = target.cuda()  # Move target masks to GPU

            # Time inference
            t0 = time.time()
            output = model(input)   # Forward pass through model
            t1 = time.time()
            total_inference_time += (t1 - t0)
            total_slices += input.size(0)

            # Compute IoU and Dice for the batch
            iou = iou_score(output, target)
            dice = dice_coef2(output, target)
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            # Compute per-slice metrics
            for i in range(input.size(0)):
                pid = pids[i]
                slice_iou = iou_score(output[i].unsqueeze(0), target[i].unsqueeze(0))
                slice_dice = dice_coef2(output[i].unsqueeze(0), target[i].unsqueeze(0))

                if pid not in per_patient_metrics_raw:
                    per_patient_metrics_raw[pid] = {"dice_vals": [], "iou_vals": []}

                per_patient_metrics_raw[pid]["dice_vals"].append(slice_dice.item())
                per_patient_metrics_raw[pid]["iou_vals"].append(slice_iou.item())

            # Post-process outputs: apply sigmoid and threshold at 0.5
            output = torch.sigmoid(output)
            output = (output > 0.5).float().cpu().numpy()
            output = np.squeeze(output, axis=1)
            raw_predictions.append(output)

            # Save outputs and Grad-CAMs
            for i in range(output.shape[0]):
                save_output(output[i:i+1], OUTPUT_MASK_DIR, test_image_paths, counter)
                save_grad_cam(output[i:i+1], GRAD_CAM_DIR, test_image_paths, counter, grad_cam)
                counter += 1

            pbar.update(1)
            pbar.set_postfix({'iou': avg_meters['iou'].avg, 'dice': avg_meters['dice'].avg})

        pbar.close()
    print("=" * 50)

    # Compute average inference time per slice
    avg_inference_ms_raw = (total_inference_time / total_slices) * 1000.0
    # print(f'Avg inference time per slice (Raw): {avg_inference_ms_raw:.2f} ms')

    # Compute unweighted (patient-level) averages
    raw_patient_dice = [np.mean(dct["dice_vals"]) for dct in per_patient_metrics_raw.values()]
    raw_patient_iou  = [np.mean(dct["iou_vals"])  for dct in per_patient_metrics_raw.values()]
    unweighted_dice_raw = np.mean(raw_patient_dice)
    unweighted_iou_raw  = np.mean(raw_patient_iou)

    print('Unweighted Raw DICE (Normal): {:.4f}'.format(unweighted_dice_raw))
    print('Unweighted Raw IoU (Normal): {:.4f}'.format(unweighted_iou_raw))

    # Calculate confusion matrix and derive performance metrics
    confusion_matrix = calculate_fp(OUTPUT_MASK_DIR, MASK_DIR, distance_threshold=args['distance_threshold'])
    tp, tn, fp, fn = confusion_matrix
    precision = calculate_precision(tp, fp)
    recall = calculate_recall(tp, fn)
    fpps = calculate_fpps(fp, total_slices)
    accuracy = calculate_accuracy(tp, tn, fp, fn)

    # Save all raw metrics into a dictionary
    metrics = OrderedDict([
        ("Dice", unweighted_dice_raw),
        ("IoU", unweighted_iou_raw),
        ("Inference Time (ms)", avg_inference_ms_raw),
        ("Total Slices", len(test_image_paths)),
        ("Total Patients", total_patients),
        ("True Positive (TP)", tp),
        ("True Negative (TN)", tn),
        ("False Positive (FP)", fp),
        ("False Negative (FN)", fn),
        ("Precision", precision),
        ("Recall", recall),
        ("FPPS", fpps),
        ("Accuracy", accuracy),
    ])

    # Save metrics to CSV file
    save_metrics_to_csv(metrics, METRICS_DIR)
    print("Raw metrics (Normal) saved.")

    #############################
    # Raw predictions for clean test set (no FPR)
    #############################

    CLEAN_NAME = 'CLEAN_' + NAME  # Create name for clean output folders

    # Define output directories for clean test set
    CLEAN_OUTPUT_MASK_DIR = os.path.join(os.getcwd(), 'model_outputs', folder, 'Segmentation_output', CLEAN_NAME)
    CLEAN_GRAD_CAM_DIR = os.path.join(os.getcwd(), 'model_outputs', folder, 'Grad_CAM_output', CLEAN_NAME)

    # Initialize average meters for clean test set
    avg_meters_clean = {'iou': AverageMeter(), 'dice': AverageMeter()}
    per_patient_metrics_clean = {}
    total_inference_time = 0.0
    total_slices = 0

    with torch.no_grad():
        counter = 0
        pbar = tqdm(total=len(clean_test_loader), desc="Raw predictions (Clean)")
        
        for (input, target, pids) in clean_test_loader:
            input = input.cuda()
            target = target.cuda()

            # Time inference
            t0 = time.time()
            output = model(input)
            t1 = time.time()
            total_inference_time += (t1 - t0)
            total_slices += input.size(0)

            # Compute IoU and Dice for the batch
            iou = iou_score(output, target)
            dice = dice_coef2(output, target)
            avg_meters_clean['iou'].update(iou, input.size(0))
            avg_meters_clean['dice'].update(dice, input.size(0))

            # Compute per-slice metrics
            for i in range(input.size(0)):
                pid = pids[i]
                slice_iou = iou_score(output[i].unsqueeze(0), target[i].unsqueeze(0))
                slice_dice = dice_coef2(output[i].unsqueeze(0), target[i].unsqueeze(0))

                if pid not in per_patient_metrics_clean:
                    per_patient_metrics_clean[pid] = {"dice_vals": [], "iou_vals": []}

                per_patient_metrics_clean[pid]["dice_vals"].append(slice_dice.item())
                per_patient_metrics_clean[pid]["iou_vals"].append(slice_iou.item())

            # Post-process outputs: apply sigmoid and threshold at 0.5
            output = torch.sigmoid(output)
            output = (output > 0.5).float().cpu().numpy()
            output = np.squeeze(output, axis=1)

            # Save outputs and Grad-CAMs
            for i in range(output.shape[0]):
                save_output(output[i:i+1], CLEAN_OUTPUT_MASK_DIR, clean_test_image_paths, counter)
                save_grad_cam(output[i:i+1], CLEAN_GRAD_CAM_DIR, clean_test_image_paths, counter, grad_cam)
                counter += 1

            pbar.update(1)

        pbar.close()
    print("=" * 50)

    # Compute average inference time per slice
    avg_inference_ms_clean = (total_inference_time / total_slices) * 1000.0

    # Compute unweighted (patient-level) averages
    clean_patient_dice = [np.mean(dct["dice_vals"]) for dct in per_patient_metrics_clean.values()]
    clean_patient_iou  = [np.mean(dct["iou_vals"])  for dct in per_patient_metrics_clean.values()]
    unweighted_dice_clean = np.mean(clean_patient_dice)
    unweighted_iou_clean  = np.mean(clean_patient_iou)

    print('Unweighted Clean DICE: {:.4f}'.format(unweighted_dice_clean))
    print('Unweighted Clean IoU: {:.4f}'.format(unweighted_iou_clean))

    # Calculate confusion matrix and derive performance metrics
    clean_confusion_matrix = calculate_fp_clean_dataset(
        os.path.join(base_dir, 'model_outputs', folder, 'Segmentation_output', CLEAN_OUTPUT_MASK_DIR)
    )
    tp_clean, tn_clean, fp_clean, fn_clean = clean_confusion_matrix
    precision_clean = calculate_precision(tp_clean, fp_clean)
    recall_clean = calculate_recall(tp_clean, fn_clean)
    fpps_clean = calculate_fpps(fp_clean, total_slices)
    accuracy_clean = calculate_accuracy(tp_clean, tn_clean, fp_clean, fn_clean)

    # Save clean metrics into a dictionary
    metrics_clean = OrderedDict([
        ("Dice", unweighted_dice_clean),
        ("IoU", unweighted_iou_clean),
        ("Inference Time (ms)", avg_inference_ms_clean),
        ("Total Slices", len(clean_test_image_paths)),
        ("Total Patients", clean_total_patients),
        ("True Positive (TP)", tp_clean),
        ("True Negative (TN)", tn_clean),
        ("False Positive (FP)", fp_clean),
        ("False Negative (FN)", fn_clean),
        ("Precision", precision_clean),
        ("Recall", recall_clean),
        ("FPPS", fpps_clean),
        ("Accuracy", accuracy_clean),
    ])

    # Save clean metrics to CSV file
    save_metrics_to_csv(metrics_clean, METRICS_DIR, filename="metrics_clean.csv")
    print("Raw metrics (Clean) saved.")

        #############################
    # Save per-patient raw and clean slice metrics
    #############################

    rows = []  # List to accumulate all per-patient results

    # Store raw patient metrics
    for pid, dct in per_patient_metrics_raw.items():
        mean_dice = float(np.mean(dct["dice_vals"]))
        mean_iou = float(np.mean(dct["iou_vals"]))
        rows.append({
            "patient_id": pid,
            "dataset_type": "raw",  # Tagging as raw dataset
            "dice_mean": mean_dice,
            "iou_mean": mean_iou
        })

    # Store clean patient metrics
    for pid, dct in per_patient_metrics_clean.items():
        mean_dice = float(np.mean(dct["dice_vals"]))
        mean_iou = float(np.mean(dct["iou_vals"]))
        rows.append({
            "patient_id": pid,
            "dataset_type": "clean",  # Tagging as clean dataset
            "dice_mean": mean_dice,
            "iou_mean": mean_iou
        })

    # Save the accumulated rows into a CSV
    df_per_patient = pd.DataFrame(rows)
    per_patient_csv = os.path.join(METRICS_DIR, "per_patient_metrics.csv")
    df_per_patient.to_csv(per_patient_csv, index=False)
    print(f"Saved per-patient slice metrics to {per_patient_csv}")

    #############################
    # Second pass: FPR post-processing on normal test set
    #############################

    # Initialize average meters for FPR post-processed results
    avg_meters_fpr = {'iou': AverageMeter(), 'dice': AverageMeter()}
    per_patient_metrics_fpr = {}  # Dictionary to accumulate per-patient metrics
    total_inference_time = 0.0
    total_slices = 0

    with torch.no_grad():
        counter = 0
        pbar = tqdm(total=len(test_loader), desc="FPR post-processing (Normal)")

        for input, target, pids in test_loader:
            input = input.cuda()
            target = target.cuda()

            # Time inference
            t0 = time.time()
            output = model(input)
            t1 = time.time()
            total_inference_time += (t1 - t0)
            total_slices += input.size(0)

            # Post-process output
            output = torch.sigmoid(output)
            output = (output > 0.5).float().cpu().numpy()
            output = np.squeeze(output, axis=1)

            batch_fpr = []  # Store FPR masks for the batch

            for j in range(output.shape[0]):
                pid = pids[j]

                # Apply FPR classifier
                fpr_mask = apply_fpr_classifier(output[j], classifier,
                                                distance_threshold=args['distance_threshold'])
                batch_fpr.append(fpr_mask)

                # Save FPR output and Grad-CAM
                save_output(fpr_mask[np.newaxis, :, :], OUTPUT_MASK_DIR, test_image_paths, counter)
                save_grad_cam(fpr_mask[np.newaxis, :, :], GRAD_CAM_DIR, test_image_paths, counter, grad_cam)

                # Compute and store per-slice metrics
                slice_iou_fpr = iou_score(torch.tensor(fpr_mask).unsqueeze(0).unsqueeze(0),
                                          target[j].unsqueeze(0).unsqueeze(0))
                slice_dice_fpr = dice_coef2(torch.tensor(fpr_mask).unsqueeze(0).unsqueeze(0),
                                            target[j].unsqueeze(0).unsqueeze(0))

                if pid not in per_patient_metrics_fpr:
                    per_patient_metrics_fpr[pid] = {"dice_vals": [], "iou_vals": []}
                per_patient_metrics_fpr[pid]["dice_vals"].append(slice_dice_fpr.item())
                per_patient_metrics_fpr[pid]["iou_vals"].append(slice_iou_fpr.item())

                counter += 1

            # Update batch metrics
            batch_fpr = np.array(batch_fpr)
            iou_fpr = iou_score(torch.tensor(batch_fpr).unsqueeze(1), target)
            dice_fpr = dice_coef2(torch.tensor(batch_fpr).unsqueeze(1), target)
            avg_meters_fpr['iou'].update(iou_fpr, input.size(0))
            avg_meters_fpr['dice'].update(dice_fpr, input.size(0))

            pbar.set_postfix({'iou': avg_meters_fpr['iou'].avg, 'dice': avg_meters_fpr['dice'].avg})
            pbar.update(1)

        pbar.close()
    print("=" * 50)

    # Calculate average inference time per slice
    avg_inference_ms_fpr = (total_inference_time / total_slices) * 1000.0

    # Compute unweighted (patient-level) averages after FPR
    fpr_patient_dice = [np.mean(dct["dice_vals"]) for dct in per_patient_metrics_fpr.values()]
    fpr_patient_iou  = [np.mean(dct["iou_vals"]) for dct in per_patient_metrics_fpr.values()]
    unweighted_dice_fpr = np.mean(fpr_patient_dice)
    unweighted_iou_fpr  = np.mean(fpr_patient_iou)

    print('Unweighted FPR DICE (Normal): {:.4f}'.format(unweighted_dice_fpr))
    print('Unweighted FPR IoU (Normal): {:.4f}'.format(unweighted_iou_fpr))

    # Calculate confusion matrix and performance metrics for FPR
    confusion_matrix_fpr = calculate_fp(OUTPUT_MASK_DIR, MASK_DIR, distance_threshold=args['distance_threshold'])
    tp_fpr, tn_fpr, fp_fpr, fn_fpr = confusion_matrix_fpr
    precision_fpr = calculate_precision(tp_fpr, fp_fpr)
    recall_fpr = calculate_recall(tp_fpr, fn_fpr)
    fpps_fpr = calculate_fpps(fp_fpr, total_slices)
    accuracy_fpr = calculate_accuracy(tp_fpr, tn_fpr, fp_fpr, fn_fpr)

    # Save FPR metrics
    metrics_fpr = OrderedDict([
        ("Dice", unweighted_dice_fpr),
        ("IoU", unweighted_iou_fpr),
        ("Inference Time (ms)", avg_inference_ms_fpr),
        ("Total Slices", len(test_image_paths)),
        ("Total Patients", total_patients),
        ("True Positive (TP)", tp_fpr),
        ("True Negative (TN)", tn_fpr),
        ("False Positive (FP)", fp_fpr),
        ("False Negative (FN)", fn_fpr),
        ("Precision", precision_fpr),
        ("Recall", recall_fpr),
        ("FPPS", fpps_fpr),
        ("Accuracy", accuracy_fpr),
    ])

    save_metrics_to_csv(metrics_fpr, METRICS_DIR, filename="metrics_fpr.csv")
    print("FPR metrics (Normal) saved.")

    #############################
    # Third pass: FPR post-processing on clean test set
    #############################

    # Initialize average meters for clean FPR results
    avg_meters_fpr_clean = {'iou': AverageMeter(), 'dice': AverageMeter()}
    per_patient_metrics_fpr_clean = {}  # Store per-patient FPR metrics
    total_inference_time = 0.0
    total_slices = 0

    with torch.no_grad():
        counter = 0
        pbar = tqdm(total=len(clean_test_loader), desc="FPR post-processing (Clean)")

        for input, target, pids in clean_test_loader:
            input = input.cuda()
            target = target.cuda()

            # Time inference
            t0 = time.time()
            output = model(input)
            t1 = time.time()
            total_inference_time += (t1 - t0)
            total_slices += input.size(0)

            # Post-process outputs
            output = torch.sigmoid(output)
            output = (output > 0.5).float().cpu().numpy()
            output = np.squeeze(output, axis=1)

            batch_fpr = []

            for j in range(output.shape[0]):
                pid = pids[j]

                # Apply FPR classifier
                fpr_mask = apply_fpr_classifier(output[j], classifier,
                                                distance_threshold=args['distance_threshold'])
                batch_fpr.append(fpr_mask)

                # Save outputs
                save_output(fpr_mask[np.newaxis, :, :], CLEAN_OUTPUT_MASK_DIR, clean_test_image_paths, counter)
                save_grad_cam(fpr_mask[np.newaxis, :, :], CLEAN_GRAD_CAM_DIR, clean_test_image_paths, counter, grad_cam)

                # Compute per-slice clean FPR metrics
                slice_iou_fpr = iou_score(torch.tensor(fpr_mask).unsqueeze(0).unsqueeze(0),
                                          target[j].unsqueeze(0).unsqueeze(0))
                slice_dice_fpr = dice_coef2(torch.tensor(fpr_mask).unsqueeze(0).unsqueeze(0),
                                            target[j].unsqueeze(0).unsqueeze(0))

                if pid not in per_patient_metrics_fpr_clean:
                    per_patient_metrics_fpr_clean[pid] = {"dice_vals": [], "iou_vals": []}
                per_patient_metrics_fpr_clean[pid]["dice_vals"].append(slice_dice_fpr.item())
                per_patient_metrics_fpr_clean[pid]["iou_vals"].append(slice_iou_fpr.item())

                counter += 1

            # Update batch metrics
            batch_fpr = np.array(batch_fpr)
            iou_fpr = iou_score(torch.tensor(batch_fpr).unsqueeze(1), target)
            dice_fpr = dice_coef2(torch.tensor(batch_fpr).unsqueeze(1), target)
            avg_meters_fpr_clean['iou'].update(iou_fpr, input.size(0))
            avg_meters_fpr_clean['dice'].update(dice_fpr, input.size(0))

            pbar.set_postfix({'iou': avg_meters_fpr_clean['iou'].avg,
                              'dice': avg_meters_fpr_clean['dice'].avg})
            pbar.update(1)

        pbar.close()
    print("=" * 50)

    # Calculate average inference time per slice
    avg_inference_ms_fpr_clean = (total_inference_time / total_slices) * 1000.0

    # Compute unweighted (patient-level) averages after FPR (clean)
    fpr_clean_patient_dice = [np.mean(dct["dice_vals"]) for dct in per_patient_metrics_fpr_clean.values()]
    fpr_clean_patient_iou  = [np.mean(dct["iou_vals"]) for dct in per_patient_metrics_fpr_clean.values()]
    unweighted_dice_fpr_clean = np.mean(fpr_clean_patient_dice)
    unweighted_iou_fpr_clean  = np.mean(fpr_clean_patient_iou)

    print('Unweighted FPR Clean DICE: {:.4f}'.format(unweighted_dice_fpr_clean))
    print('Unweighted FPR Clean IoU: {:.4f}'.format(unweighted_iou_fpr_clean))

    # Calculate clean confusion matrix and metrics
    clean_confusion_matrix_fpr = calculate_fp_clean_dataset(
        os.path.join(base_dir, 'model_outputs', folder, 'Segmentation_output', CLEAN_OUTPUT_MASK_DIR)
    )
    tp_clean_fpr, tn_clean_fpr, fp_clean_fpr, fn_clean_fpr = clean_confusion_matrix_fpr
    precision_clean_fpr = calculate_precision(tp_clean_fpr, fp_clean_fpr)
    recall_clean_fpr = calculate_recall(tp_clean_fpr, fn_clean_fpr)
    fpps_clean_fpr = calculate_fpps(fp_clean_fpr, total_slices)
    accuracy_clean_fpr = calculate_accuracy(tp_clean_fpr, tn_clean_fpr, fp_clean_fpr, fn_clean_fpr)

    # Save clean FPR metrics
    metrics_fpr_clean = OrderedDict([
        ("Dice", unweighted_dice_fpr_clean),
        ("IoU", unweighted_iou_fpr_clean),
        ("Inference Time (ms)", avg_inference_ms_fpr_clean),
        ("Total Slices", len(clean_test_image_paths)),
        ("Total Patients", clean_total_patients),
        ("True Positive (TP)", tp_clean_fpr),
        ("True Negative (TN)", tn_clean_fpr),
        ("False Positive (FP)", fp_clean_fpr),
        ("False Negative (FN)", fn_clean_fpr),
        ("Precision", precision_clean_fpr),
        ("Recall", recall_clean_fpr),
        ("FPPS", fpps_clean_fpr),
        ("Accuracy", accuracy_clean_fpr),
    ])

    save_metrics_to_csv(metrics_fpr_clean, METRICS_DIR, filename="metrics_fpr_clean.csv")
    print("FPR metrics (Clean) saved.")

    #############################
    # Merge and save per-patient FPR slice metrics
    #############################

    rows_fpr = []  # List to accumulate FPR per-patient results

    # Merge raw FPR patient metrics
    for pid, dct in per_patient_metrics_fpr.items():
        mean_dice = float(np.mean(dct["dice_vals"]))
        mean_iou = float(np.mean(dct["iou_vals"]))
        rows_fpr.append({
            "patient_id": pid,
            "dataset_type": "raw_fpr",  # Tagging as raw FPR dataset
            "dice_mean": mean_dice,
            "iou_mean": mean_iou
        })

    # Merge clean FPR patient metrics
    for pid, dct in per_patient_metrics_fpr_clean.items():
        mean_dice = float(np.mean(dct["dice_vals"]))
        mean_iou = float(np.mean(dct["iou_vals"]))
        rows_fpr.append({
            "patient_id": pid,
            "dataset_type": "clean_fpr",  # Tagging as clean FPR dataset
            "dice_mean": mean_dice,
            "iou_mean": mean_iou
        })

    # Save the merged FPR metrics into a CSV
    df_per_patient_fpr = pd.DataFrame(rows_fpr)
    per_patient_fpr_csv = os.path.join(METRICS_DIR, "per_patient_metrics_fpr.csv")
    df_per_patient_fpr.to_csv(per_patient_fpr_csv, index=False)
    print(f"Saved merged per-patient FPR metrics to {per_patient_fpr_csv}")

#############################
# Run the main function
#############################

if __name__ == '__main__':
    main()
