#!/usr/bin/env python
import argparse
import os
import pandas as pd
import numpy as np
import torch
import yaml
import joblib
from tqdm import tqdm
import xgboost as xgb
from collections import OrderedDict

# For morphological feature extraction and connectivity-based analysis
from skimage.measure import regionprops
from scipy.ndimage import center_of_mass, generate_binary_structure, label as ndi_label

# Import your model architectures (adjust these imports as needed)
from Unet.unet_model import UNet
from UnetNested.Nested_Unet import NestedUNet

# -------------------------
# Helper functions
# -------------------------
def is_true_detection(pred_mask, gt_mask, distance_threshold=80):
    """
    Determines if a predicted mask represents a true detection (returns 1)
    or a false positive (returns 0) using a connectivity-based center-of-mass method.
    
    For a non-clean image (where gt_mask is nonempty), compute the center of mass
    of gt_mask. Then, label connected regions in pred_mask. If at least one regions
    center of mass is within the distance_threshold (in pixels) of the ground truth
    center, return 1; otherwise, return 0.
    """
    if np.sum(gt_mask) == 0:
        # Not intended for clean images.
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
    Extracts morphological features from a binary mask:
      - area: number of pixels
      - perimeter: boundary length (fallback to 1 if zero)
      - eccentricity: measure of elongation
      - solidity: ratio of area to convex area
      - compactness: area divided by (perimeter^2)
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

# -------------------------
# Main processing
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', required=True,
                        help='Folder containing model outputs (config.yml and model.pth)')
    parser.add_argument('--model_name', default='UNET', choices=['UNET', 'NestedUNET'],
                        help='Model architecture used')
    parser.add_argument('--augmentation', default=False, type=lambda s: s.lower()=='true',
                        help='Whether augmentation was used (affects naming)')
    parser.add_argument('--distance_threshold', type=float, default=80,
                        help='Distance threshold (in pixels) for matching centers of mass')
    args = parser.parse_args()

    base_dir = os.getcwd()
    config_path = os.path.join(base_dir, 'model_outputs', args.folder, 'config.yml')
    model_path = os.path.join(base_dir, 'model_outputs', args.folder, 'model.pth')

    # Create output directory for FP classifier results inside the given folder
    fp_out_dir = os.path.join(base_dir, 'model_outputs', args.folder, 'fp_classifier')
    os.makedirs(fp_out_dir, exist_ok=True)

    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create and load the segmentation model (consistent with validate.py)
    if args.model_name == 'NestedUNET':
        model = NestedUNet(num_classes=1)
    else:
        model = UNet(n_channels=1, n_classes=1, bilinear=True)
    # Set weights_only=True to address FutureWarning
    state_dict = torch.load(model_path, weights_only=True, map_location='cuda')
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

    # Fixed directories (consistent with validate.py)
    IMAGE_DIR = '/dcs/22/u2202609/year_3/cs310/Project/Preprocessing/data/Image/'
    MASK_DIR = '/dcs/22/u2202609/year_3/cs310/Project/Preprocessing/data/Mask/'
    CLEAN_DIR_IMG = '/dcs/22/u2202609/year_3/cs310/Project/Preprocessing/data/Clean/Image/'
    CLEAN_DIR_MASK = '/dcs/22/u2202609/year_3/cs310/Project/Preprocessing/data/Clean/Mask/'

    # Load meta CSVs for normal and clean cases
    meta_csv = os.path.join(base_dir, '..', 'Preprocessing', 'csv', 'meta.csv')
    clean_meta_csv = os.path.join(base_dir, '..', 'Preprocessing', 'csv', 'clean_meta.csv')
    
    meta = pd.read_csv(meta_csv)
    clean_meta = pd.read_csv(clean_meta_csv)
    
    # Prepend file paths (as in validate.py)
    meta['original_image'] = meta['original_image'].apply(lambda x: IMAGE_DIR + x + '.npy')
    meta['mask_image'] = meta['mask_image'].apply(lambda x: MASK_DIR + x + '.npy')
    clean_meta['original_image'] = clean_meta['original_image'].apply(lambda x: CLEAN_DIR_IMG + x + '.npy')
    clean_meta['mask_image'] = clean_meta['mask_image'].apply(lambda x: CLEAN_DIR_MASK + x + '.npy')
    
    # Combine training and validation samples (ignore the meta.csv label)
    normal_train = meta[meta['data_split'].isin(['Train', 'Validation'])].copy()
    clean_train = clean_meta[clean_meta['data_split'].isin(['Train', 'Validation'])].copy()

    features_list = []
    
    # Process normal (nodule) training examples
    for idx, row in tqdm(normal_train.iterrows(), total=len(normal_train), desc="Normal set features"):
        image_path = row['original_image']
        gt_mask_path = row['mask_image']
        try:
            image = np.load(image_path)
            gt_mask = np.load(gt_mask_path)
        except Exception as e:
            print(f"Error loading {image_path} or {gt_mask_path}: {e}")
            continue

        image_tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0).float().cuda()
        with torch.no_grad():
            output = model(image_tensor)
            output = torch.sigmoid(output)
            pred_mask = (output > 0.5).float().cpu().numpy()[0, 0]

        # Use the connectivity-based method for non-clean images
        detection_label = is_true_detection(pred_mask, gt_mask, distance_threshold=args.distance_threshold)
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
    
    # Process clean training examples (should contain no nodules)
    for idx, row in tqdm(clean_train.iterrows(), total=len(clean_train), desc="Clean set features"):
        image_path = row['original_image']
        gt_mask_path = row['mask_image']
        try:
            image = np.load(image_path)
            gt_mask = np.load(gt_mask_path)
        except Exception as e:
            print(f"Error loading {image_path} or {gt_mask_path}: {e}")
            continue

        image_tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0).float().cuda()
        with torch.no_grad():
            output = model(image_tensor)
            output = torch.sigmoid(output)
            pred_mask = (output > 0.5).float().cpu().numpy()[0, 0]

        # For a clean image, if the predicted mask is empty, skip the sample;
        # otherwise, label as false positive (0)
        if np.sum(pred_mask) == 0:
            continue
        else:
            detection_label = 0

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

    # -------------------------
    # Train XGBoost Classifier for False Positive Reduction
    # -------------------------
    feature_columns = [
        'malignancy', 'subtlety', 'texture', 'sphericity', 'margin',
        'area', 'perimeter', 'eccentricity', 'solidity', 'compactness'
    ]
    X = features_df[feature_columns].values
    y = features_df['label'].values

    from sklearn.model_selection import KFold, cross_validate
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

if __name__ == '__main__':
    main()
