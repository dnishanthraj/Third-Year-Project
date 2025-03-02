#!/usr/bin/env python
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
from tqdm import tqdm
import xgboost as xgb
import joblib

# For morphological features and connectivity analysis
from skimage.measure import regionprops, label as sklabel
from scipy.ndimage import center_of_mass, generate_binary_structure, label as ndi_label

# Import your model architectures (adjust these imports as needed)
from Unet.unet_model import UNet
from UnetNested.Nested_Unet import NestedUNet  # Note: changed variable name to NestedUNET to match below usage

# Import local modules
from dataset import MyLidcDataset
from metrics import (iou_score, dice_coef2, calculate_fp, calculate_fp_clean_dataset, calculate_accuracy, calculate_f1_score, calculate_specificity, 
                     calculate_precision, calculate_recall, calculate_fpps, save_metrics_to_csv)
from utils import AverageMeter, str2bool
from grad_cam import GradCAM
from sklearn.model_selection import KFold, cross_validate

#############################
# Helper Functions
#############################

def is_true_detection(pred_mask, gt_mask, distance_threshold=80):
    """For a non-clean image, returns 1 if any connected candidate's center-of-mass is within the threshold of the ground truth center; else 0."""
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
    """Extracts [area, perimeter, eccentricity, solidity, compactness] from a binary mask."""
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
    file_label = test_image_paths[counter][-23:].replace('NI', 'PD').replace('.npy', '.npy')
    save_path = os.path.join(output_directory, file_label)
    os.makedirs(output_directory, exist_ok=True)
    np.save(save_path, output[0, :, :])

def save_grad_cam(output, grad_cam_dir, test_image_paths, counter, grad_cam_generator):
    file_label = test_image_paths[counter][-23:].replace('NI', 'GC').replace('.npy', '.npy')
    with torch.set_grad_enabled(True):
        heatmap = grad_cam_generator.generate(
            torch.tensor(output[0, :, :]).unsqueeze(0).unsqueeze(0).cuda(),
            class_idx=0
        )
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    grad_cam_save_path = os.path.join(grad_cam_dir, file_label)
    os.makedirs(grad_cam_dir, exist_ok=True)
    np.save(grad_cam_save_path, heatmap)

def apply_fpr_classifier(pred_mask, clinical_feats, classifier, distance_threshold=80):
    """
    For each connected candidate in pred_mask, extracts morphological features and forms a feature vector
    (clinical_feats + morphological features). Uses the classifier to predict a label.
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
        feature_vector = np.array(clinical_feats + morph_feats).reshape(1, -1)
        pred_label = classifier.predict(feature_vector)
        if pred_label[0] == 0:
            modified_mask[candidate == 1] = 0
    return modified_mask

#############################
# FP Classifier Training (if needed)
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
        clinical_feats = [
            row.get('malignancy', 0),
            row.get('subtlety', 0),
            row.get('texture', 0),
            row.get('sphericity', 0),
            row.get('margin', 0)
        ]
        feat_dict = {
            'malignancy': clinical_feats[0],
            'subtlety': clinical_feats[1],
            'texture': clinical_feats[2],
            'sphericity': clinical_feats[3],
            'margin': clinical_feats[4],
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
        clinical_feats = [
            row.get('malignancy', 0),
            row.get('subtlety', 0),
            row.get('texture', 0),
            row.get('sphericity', 0),
            row.get('margin', 0)
        ]
        feat_dict = {
            'malignancy': clinical_feats[0],
            'subtlety': clinical_feats[1],
            'texture': clinical_feats[2],
            'sphericity': clinical_feats[3],
            'margin': clinical_feats[4],
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
    
    feature_columns = ['malignancy', 'subtlety', 'texture', 'sphericity', 'margin',
                       'area', 'perimeter', 'eccentricity', 'solidity', 'compactness']
    X = features_df[feature_columns].values
    y = features_df['label'].values

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
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
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default="UNET", choices=['UNET', 'NestedUNET'],
                        help='Model name (UNET or NestedUNET)')
    parser.add_argument('--augmentation', default=False, type=str2bool,
                        help='Whether augmentation was used')
    parser.add_argument('--folder', required=True,
                        help='Folder containing model and config files')
    parser.add_argument('--distance_threshold', type=float, default=80,
                        help='Distance threshold for FPR classifier')
    return parser.parse_args()

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
    # FPR_OUTPUT_MASK_DIR = os.path.join(base_dir, 'model_outputs', folder, 'Segmentation_output', NAME + '_fpr')
    # FPR_GRAD_CAM_DIR = os.path.join(base_dir, 'model_outputs', folder, 'Grad_CAM_output', NAME + '_fpr')
    # os.makedirs(FPR_OUTPUT_MASK_DIR, exist_ok=True)
    # os.makedirs(FPR_GRAD_CAM_DIR, exist_ok=True)
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

    # Load Grad-CAM generator
    grad_cam = GradCAM(model, model.conv2_0)

    # Fixed directories for data
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

    # Build clinical features dictionaries (keyed by the last 23 characters of the image path)
    clinical_dict = {}
    for idx, row in test_meta.iterrows():
        identifier = row['original_image'][-23:]
        clinical_feats = [
            row.get('malignancy', 0),
            row.get('subtlety', 0),
            row.get('texture', 0),
            row.get('sphericity', 0),
            row.get('margin', 0)
        ]
        clinical_dict[identifier] = clinical_feats

    clean_clinical_dict = {}
    for idx, row in clean_test_meta.iterrows():
        identifier = row['original_image'][-23:]
        clean_clinical_dict[identifier] = [
            row.get('malignancy', 0),
            row.get('subtlety', 0),
            row.get('texture', 0),
            row.get('sphericity', 0),
            row.get('margin', 0)
        ]

    # Load test datasets
    test_image_paths = list(test_meta['original_image'])
    test_mask_paths = list(test_meta['mask_image'])
    total_patients = len(test_meta.groupby('patient_id'))

    clean_test_image_paths = list(clean_test_meta['original_image'])
    clean_test_mask_paths = list(clean_test_meta['mask_image'])
    clean_total_patients = len(clean_test_meta.groupby('patient_id'))

    test_dataset = MyLidcDataset(test_image_paths, test_mask_paths, return_pid=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'],
                                                shuffle=False, pin_memory=True, drop_last=False, num_workers=12)
    clean_test_dataset = MyLidcDataset(clean_test_image_paths, clean_test_mask_paths, return_pid=True)
    clean_test_loader = torch.utils.data.DataLoader(clean_test_dataset, batch_size=config['batch_size'],
                                                     shuffle=False, pin_memory=True, drop_last=False, num_workers=12)

    # -------------------------
    # Check for FP classifier; if not found, train it.
    # -------------------------
    classifier_path = os.path.join(fp_out_dir, "xgb_fpr_model.pkl")
    if not os.path.exists(classifier_path):
        meta_csv = os.path.join(base_dir, '..', 'Preprocessing', 'csv', 'meta.csv')
        clean_meta_csv = os.path.join(base_dir, '..', 'Preprocessing', 'csv', 'clean_meta.csv')
        classifier = train_fpr_classifier(fp_out_dir, meta_csv, clean_meta_csv,
                                          IMAGE_DIR, MASK_DIR, CLEAN_DIR_IMG, CLEAN_DIR_MASK, model)
    else:
        classifier = joblib.load(classifier_path)

    #############################
    # First pass: Raw predictions for normal test set
    #############################
    avg_meters = {'iou': AverageMeter(), 'dice': AverageMeter()}
    per_patient_metrics_raw = {}
    dice_list = []
    iou_list = []

    with torch.no_grad():
        counter = 0
        pbar = tqdm(total=len(test_loader), desc="Raw predictions (Normal)")
        raw_predictions = []
        for (input, target, pids) in test_loader:
            input = input.cuda()
            target = target.cuda()
            output = model(input)
            iou = iou_score(output, target)
            dice = dice_coef2(output, target)
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            for i in range(input.size(0)):
                pid = pids[i]
                slice_iou = iou_score(output[i].unsqueeze(0), target[i].unsqueeze(0))
                slice_dice = dice_coef2(output[i].unsqueeze(0), target[i].unsqueeze(0))
                dice_list.append(slice_dice)
                iou_list.append(slice_iou)

                if pid not in per_patient_metrics_raw:
                    per_patient_metrics_raw[pid] = {"dice_vals": [], "iou_vals": []}

                per_patient_metrics_raw[pid]["dice_vals"].append(slice_dice.item())
                per_patient_metrics_raw[pid]["iou_vals"].append(slice_iou.item())

            output = torch.sigmoid(output)
            output = (output > 0.5).float().cpu().numpy()
            output = np.squeeze(output, axis=1)
            raw_predictions.append(output)
            for i in range(output.shape[0]):
                save_output(output[i:i+1], OUTPUT_MASK_DIR, test_image_paths, counter)
                save_grad_cam(output[i:i+1], GRAD_CAM_DIR, test_image_paths, counter, grad_cam)
                counter += 1
            pbar.update(1)
            pbar.set_postfix({'iou': avg_meters['iou'].avg, 'dice': avg_meters['dice'].avg})
        pbar.close()
    print("=" * 50)
    print('Raw IoU (Normal): {:.4f}'.format(avg_meters['iou'].avg))
    print('Raw DICE (Normal): {:.4f}'.format(avg_meters['dice'].avg))
    
    confusion_matrix = calculate_fp(OUTPUT_MASK_DIR, MASK_DIR, distance_threshold=args['distance_threshold'])
    tp, tn, fp, fn = confusion_matrix
    precision = calculate_precision(tp, fp)
    recall = calculate_recall(tp, fn)
    fpps = calculate_fpps(fp, total_patients)
    accuracy = calculate_accuracy(tp, tn, fp, fn)
    specificity = calculate_specificity(tn, fp)
    f1_score = calculate_f1_score(precision, recall)
    dice_std = np.std(dice_list)
    iou_std = np.std(iou_list)

    metrics = OrderedDict([
        ("Dice", avg_meters['dice'].avg),
        ("Dice_std", dice_std),
        ("IoU", avg_meters['iou'].avg),
         ("IoU_std", iou_std),
        ("Total Slices", len(test_image_paths)),
        ("Total Patients", total_patients),
        ("True Positive (TP)", tp),
        ("True Negative (TN)", tn),
        ("False Positive (FP)", fp),
        ("False Negative (FN)", fn),
        ("Precision", precision),
        ("Recall", recall),
        ("FPPS", fpps),
        ("Accuracy", accuracy),  # Add Accuracy
        ("Specificity", specificity),  # Add Specificity
        ("F1-Score", f1_score)  # Add F1-Score
    ])
    save_metrics_to_csv(metrics, METRICS_DIR)
    print("Raw metrics (Normal) saved.")

    CLEAN_NAME = 'CLEAN_'+NAME

    # CLEAN_OUTPUT_MASK_DIR = '/home/LUNG_DATA/Segmentation_output/{}'.format(CLEAN_NAME)
    # CLEAN_OUTPUT_MASK_DIR = '/dcs/22/u2202609/year_3/cs310/Project/Segmentation/Segmentation_output/{}'.format(CLEAN_NAME)
    CLEAN_OUTPUT_MASK_DIR = os.path.join(os.getcwd(), 'model_outputs', folder, 'Segmentation_output', CLEAN_NAME)
    CLEAN_GRAD_CAM_DIR = os.path.join(os.getcwd(), 'model_outputs', folder, 'Grad_CAM_output', CLEAN_NAME)

    #############################
    # New Block: Raw predictions for clean test set (without FPR)
    #############################
    avg_meters_clean = {'iou': AverageMeter(), 'dice': AverageMeter()}
    per_patient_metrics_clean = {}
    dice_list = []
    iou_list = []

    with torch.no_grad():
        counter = 0
        pbar = tqdm(total=len(clean_test_loader), desc="Raw predictions (Clean)")
        for (input, target, pids) in clean_test_loader:
            input = input.cuda()
            target = target.cuda()
            output = model(input)
            iou = iou_score(output, target)
            dice = dice_coef2(output, target)
            avg_meters_clean['iou'].update(iou, input.size(0))
            avg_meters_clean['dice'].update(dice, input.size(0))

            for i in range(input.size(0)):
                pid = pids[i]
                slice_iou = iou_score(output[i].unsqueeze(0), target[i].unsqueeze(0))
                slice_dice = dice_coef2(output[i].unsqueeze(0), target[i].unsqueeze(0))
                dice_list.append(slice_dice)
                iou_list.append(slice_iou)

                if pid not in per_patient_metrics_clean:
                    per_patient_metrics_clean[pid] = {"dice_vals": [], "iou_vals": []}

                per_patient_metrics_clean[pid]["dice_vals"].append(slice_dice.item())
                per_patient_metrics_clean[pid]["iou_vals"].append(slice_iou.item())

            output = torch.sigmoid(output)
            output = (output > 0.5).float().cpu().numpy()
            output = np.squeeze(output, axis=1)
            for i in range(output.shape[0]):
                save_output(output[i:i+1], CLEAN_OUTPUT_MASK_DIR, clean_test_image_paths, counter)
                save_grad_cam(output[i:i+1], CLEAN_GRAD_CAM_DIR, clean_test_image_paths, counter, grad_cam)
                counter += 1
            pbar.update(1)
        pbar.close()
    print("=" * 50)
    print('Raw IoU (Clean): {:.4f}'.format(avg_meters_clean['iou'].avg))
    print('Raw DICE (Clean): {:.4f}'.format(avg_meters_clean['dice'].avg))

    # Compute unweighted (patient-level) averages for the raw test set
    raw_patient_dice = [np.mean(dct["dice_vals"]) for dct in per_patient_metrics_raw.values()]
    raw_patient_iou  = [np.mean(dct["iou_vals"])  for dct in per_patient_metrics_raw.values()]
    unweighted_dice_raw = np.mean(raw_patient_dice)
    unweighted_iou_raw  = np.mean(raw_patient_iou)

    print('Unweighted Raw DICE (Normal): {:.4f}'.format(unweighted_dice_raw))
    print('Unweighted Raw IoU (Normal): {:.4f}'.format(unweighted_iou_raw))

    clean_confusion_matrix = calculate_fp_clean_dataset(os.path.join(base_dir, 'model_outputs', folder, 'Segmentation_output', CLEAN_OUTPUT_MASK_DIR))
    tp_clean, tn_clean, fp_clean, fn_clean = clean_confusion_matrix
    precision_clean = calculate_precision(tp_clean, fp_clean)
    recall_clean = calculate_recall(tp_clean, fn_clean)
    fpps_clean = calculate_fpps(fp_clean, clean_total_patients)
    accuracy_clean = calculate_accuracy(tp_clean, tn_clean, fp_clean, fn_clean)
    specificity_clean = calculate_specificity(tn_clean, fp_clean)
    f1_score_clean = calculate_f1_score(precision_clean, recall_clean)
    
    metrics_clean = OrderedDict([
        ("Dice", avg_meters_clean['dice'].avg),
        ("IoU", avg_meters_clean['iou'].avg),
        ("Total Slices", len(clean_test_image_paths)),
        ("Total Patients", clean_total_patients),
        ("True Positive (TP)", tp_clean),
        ("True Negative (TN)", tn_clean),
        ("False Positive (FP)", fp_clean),
        ("False Negative (FN)", fn_clean),
        ("Precision", precision_clean),
        ("Recall", recall_clean),
        ("FPPS", fpps_clean),
        ("Accuracy", accuracy_clean),  # Add Accuracy
        ("Specificity", specificity_clean),  # Add Specificity
        ("F1-Score", f1_score_clean)  # Add F1-Score
    ])
    save_metrics_to_csv(metrics_clean, METRICS_DIR, filename="metrics_clean.csv")
    print("Raw metrics (Clean) saved.")

    rows = []

    # Store raw patients
    for pid, dct in per_patient_metrics_raw.items():
        mean_dice = float(np.mean(dct["dice_vals"]))
        mean_iou = float(np.mean(dct["iou_vals"]))
        rows.append({
            "patient_id": pid,
            "dataset_type": "raw",  # NEW COLUMN
            "dice_mean": mean_dice,
            "iou_mean": mean_iou
        })

    # Store clean patients
    for pid, dct in per_patient_metrics_clean.items():
        mean_dice = float(np.mean(dct["dice_vals"]))
        mean_iou = float(np.mean(dct["iou_vals"]))
        rows.append({
            "patient_id": pid,
            "dataset_type": "clean",  # NEW COLUMN
            "dice_mean": mean_dice,
            "iou_mean": mean_iou
        })

    # Save final CSV
    df_per_patient = pd.DataFrame(rows)
    per_patient_csv = os.path.join(METRICS_DIR, "per_patient_metrics.csv")
    df_per_patient.to_csv(per_patient_csv, index=False)
    print(f"Saved per-patient slice metrics to {per_patient_csv}")


    #############################
    # Second pass: FPR post-processing on normal test set
    #############################
    avg_meters_fpr = {'iou': AverageMeter(), 'dice': AverageMeter()}
    per_patient_metrics_fpr = {}
    with torch.no_grad():
        counter = 0
        pbar = tqdm(total=len(test_loader), desc="FPR post-processing (Normal)")
        for input, target, pids in test_loader:
            input = input.cuda()
            target = target.cuda()
            output = model(input)
            output = torch.sigmoid(output)
            output = (output > 0.5).float().cpu().numpy()
            output = np.squeeze(output, axis=1)
            batch_fpr = []
            for j in range(output.shape[0]):
                pid = pids[j]  # Get the patient ID for this slice
                identifier = test_image_paths[counter][-23:]
                clinical_feats = clinical_dict.get(identifier, [0, 0, 0, 0, 0])
                fpr_mask = apply_fpr_classifier(output[j], clinical_feats, classifier,
                                                distance_threshold=args['distance_threshold'])
                batch_fpr.append(fpr_mask)
                save_output(fpr_mask[np.newaxis, :, :], OUTPUT_MASK_DIR, test_image_paths, counter)
                save_grad_cam(fpr_mask[np.newaxis, :, :], GRAD_CAM_DIR, test_image_paths, counter, grad_cam)

                # Compute per-slice metrics for FPR output
                slice_iou_fpr = iou_score(torch.tensor(fpr_mask).unsqueeze(0).unsqueeze(0), target[j].unsqueeze(0).unsqueeze(0))
                slice_dice_fpr = dice_coef2(torch.tensor(fpr_mask).unsqueeze(0).unsqueeze(0), target[j].unsqueeze(0).unsqueeze(0))
                
                # Store in per_patient_metrics_fpr dictionary
                if pid not in per_patient_metrics_fpr:
                    per_patient_metrics_fpr[pid] = {"dice_vals": [], "iou_vals": []}
                per_patient_metrics_fpr[pid]["dice_vals"].append(slice_dice_fpr.item())
                per_patient_metrics_fpr[pid]["iou_vals"].append(slice_iou_fpr.item())

                counter += 1

            batch_fpr = np.array(batch_fpr)
            iou_fpr = iou_score(torch.tensor(batch_fpr).unsqueeze(1), target)
            dice_fpr = dice_coef2(torch.tensor(batch_fpr).unsqueeze(1), target)
            avg_meters_fpr['iou'].update(iou_fpr, input.size(0))
            avg_meters_fpr['dice'].update(dice_fpr, input.size(0))
            pbar.set_postfix({'iou': avg_meters_fpr['iou'].avg, 'dice': avg_meters_fpr['dice'].avg})
            pbar.update(1)
        pbar.close()
    print("=" * 50)
    print('FPR IoU (Normal): {:.4f}'.format(avg_meters_fpr['iou'].avg))
    print('FPR DICE (Normal): {:.4f}'.format(avg_meters_fpr['dice'].avg))
    confusion_matrix_fpr = calculate_fp(OUTPUT_MASK_DIR, MASK_DIR, distance_threshold=args['distance_threshold'])
    tp_fpr, tn_fpr, fp_fpr, fn_fpr = confusion_matrix_fpr
    precision_fpr = calculate_precision(tp_fpr, fp_fpr)
    recall_fpr = calculate_recall(tp_fpr, fn_fpr)
    fpps_fpr = calculate_fpps(fp_fpr, total_patients)
    accuracy_fpr = calculate_accuracy(tp_fpr, tn_fpr, fp_fpr, fn_fpr)
    specificity_fpr = calculate_specificity(tn_fpr, fp_fpr)
    f1_score_fpr = calculate_f1_score(precision_fpr, recall_fpr)
    metrics_fpr = OrderedDict([
        ("Dice", avg_meters_fpr['dice'].avg),
        ("IoU", avg_meters_fpr['iou'].avg),
        ("Total Slices", len(test_image_paths)),
        ("Total Patients", total_patients),
        ("True Positive (TP)", tp_fpr),
        ("True Negative (TN)", tn_fpr),
        ("False Positive (FP)", fp_fpr),
        ("False Negative (FN)", fn_fpr),
        ("Precision", precision_fpr),
        ("Recall", recall_fpr),
        ("FPPS", fpps_fpr),
        ("Accuracy", accuracy_fpr),  # Add Accuracy
        ("Specificity", specificity_fpr),  # Add Specificity
        ("F1-Score", f1_score_fpr)  # Add F1-Score
    ])
    save_metrics_to_csv(metrics_fpr, METRICS_DIR, filename="metrics_fpr.csv")
    print("FPR metrics (Normal) saved.")

        #############################
    # Third pass: FPR post-processing on clean test set
    #############################
    avg_meters_fpr_clean = {'iou': AverageMeter(), 'dice': AverageMeter()}
    per_patient_metrics_fpr_clean = {}  # Initialize a separate dictionary for clean FPR metrics
    with torch.no_grad():
        counter = 0
        pbar = tqdm(total=len(clean_test_loader), desc="FPR post-processing (Clean)")
        for input, target, pids in clean_test_loader:
            input = input.cuda()
            target = target.cuda()
            output = model(input)
            output = torch.sigmoid(output)
            output = (output > 0.5).float().cpu().numpy()
            output = np.squeeze(output, axis=1)
            batch_fpr = []
            for j in range(output.shape[0]):
                pid = pids[j]  # Extract patient ID for this slice
                identifier = clean_test_image_paths[counter][-23:]
                clinical_feats = clean_clinical_dict.get(identifier, [0, 0, 0, 0, 0])
                fpr_mask = apply_fpr_classifier(output[j], clinical_feats, classifier,
                                                distance_threshold=args['distance_threshold'])
                batch_fpr.append(fpr_mask)
                save_output(fpr_mask[np.newaxis, :, :], CLEAN_OUTPUT_MASK_DIR, clean_test_image_paths, counter)
                save_grad_cam(fpr_mask[np.newaxis, :, :], CLEAN_GRAD_CAM_DIR, clean_test_image_paths, counter, grad_cam)
                
                # Compute per-slice metrics for clean FPR
                slice_iou_fpr = iou_score(torch.tensor(fpr_mask).unsqueeze(0).unsqueeze(0),
                                           target[j].unsqueeze(0).unsqueeze(0))
                slice_dice_fpr = dice_coef2(torch.tensor(fpr_mask).unsqueeze(0).unsqueeze(0),
                                             target[j].unsqueeze(0).unsqueeze(0))
                
                if pid not in per_patient_metrics_fpr_clean:
                    per_patient_metrics_fpr_clean[pid] = {"dice_vals": [], "iou_vals": []}
                per_patient_metrics_fpr_clean[pid]["dice_vals"].append(slice_dice_fpr.item())
                per_patient_metrics_fpr_clean[pid]["iou_vals"].append(slice_iou_fpr.item())
                
                counter += 1
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
    print('FPR IoU (Clean): {:.4f}'.format(avg_meters_fpr_clean['iou'].avg))
    print('FPR DICE (Clean): {:.4f}'.format(avg_meters_fpr_clean['dice'].avg))

    clean_confusion_matrix_fpr = calculate_fp_clean_dataset(
        os.path.join(base_dir, 'model_outputs', folder, 'Segmentation_output', CLEAN_OUTPUT_MASK_DIR)
    )
    tp_clean_fpr, tn_clean_fpr, fp_clean_fpr, fn_clean_fpr = clean_confusion_matrix_fpr
    precision_clean_fpr = calculate_precision(tp_clean_fpr, fp_clean_fpr)
    recall_clean_fpr = calculate_recall(tp_clean_fpr, fn_clean_fpr)
    fpps_clean_fpr = calculate_fpps(fp_clean_fpr, clean_total_patients)
    accuracy_clean_fpr = calculate_accuracy(tp_clean_fpr, tn_clean_fpr, fp_clean_fpr, fn_clean_fpr)
    specificity_clean_fpr = calculate_specificity(tn_clean_fpr, fp_clean_fpr)
    f1_score_clean_fpr = calculate_f1_score(precision_clean_fpr, recall_clean_fpr)
    
    metrics_fpr_clean = OrderedDict([
        ("Dice", avg_meters_fpr_clean['dice'].avg),
        ("IoU", avg_meters_fpr_clean['iou'].avg),
        ("Total Slices", len(clean_test_image_paths)),
        ("Total Patients", clean_total_patients),
        ("True Positive (TP)", tp_clean_fpr),
        ("True Negative (TN)", tn_clean_fpr),
        ("False Positive (FP)", fp_clean_fpr),
        ("False Negative (FN)", fn_clean_fpr),
        ("Precision", precision_clean_fpr),
        ("Recall", recall_clean_fpr),
        ("FPPS", fpps_clean_fpr),
        ("Accuracy", accuracy_clean_fpr),  # Add Accuracy
        ("Specificity", specificity_clean_fpr),  # Add Specificity
        ("F1-Score", f1_score_clean_fpr)  # Add F1-Score
    ])
    save_metrics_to_csv(metrics_fpr_clean, METRICS_DIR, filename="metrics_fpr_clean.csv")
    print("FPR metrics (Clean) saved.")

    rows_fpr = []

    # Merge raw FPR metrics
    for pid, dct in per_patient_metrics_fpr.items():
        mean_dice = float(np.mean(dct["dice_vals"]))
        mean_iou = float(np.mean(dct["iou_vals"]))
        rows_fpr.append({
            "patient_id": pid,
            "dataset_type": "raw_fpr",
            "dice_mean": mean_dice,
            "iou_mean": mean_iou
        })

    # Merge clean FPR metrics
    for pid, dct in per_patient_metrics_fpr_clean.items():
        mean_dice = float(np.mean(dct["dice_vals"]))
        mean_iou = float(np.mean(dct["iou_vals"]))
        rows_fpr.append({
            "patient_id": pid,
            "dataset_type": "clean_fpr",
            "dice_mean": mean_dice,
            "iou_mean": mean_iou
        })

    df_per_patient_fpr = pd.DataFrame(rows_fpr)
    per_patient_fpr_csv = os.path.join(METRICS_DIR, "per_patient_metrics_fpr.csv")
    df_per_patient_fpr.to_csv(per_patient_fpr_csv, index=False)
    print(f"Saved merged per-patient FPR metrics to {per_patient_fpr_csv}")


if __name__ == '__main__':
    main()
