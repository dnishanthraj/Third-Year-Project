# ---------------------------------------------------------
# Metrics Computation for Segmentation Evaluation
# ---------------------------------------------------------

import numpy as np
import torch
from scipy.ndimage import label, generate_binary_structure
import torch.nn.functional as F
import os
from scipy import ndimage as ndi

# ------------------------
# Basic Metrics
# ------------------------

def iou_score(output, target):
    """
    Computes Intersection over Union (IoU) score between output and target masks.

    Args:
        output (torch.Tensor or np.ndarray): Predicted mask.
        target (torch.Tensor or np.ndarray): Ground truth mask.

    Returns:
        float: IoU score.
    """
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


def dice_coef(output, target):
    """
    Computes Dice coefficient between predicted and ground truth masks.

    Args:
        output (torch.Tensor): Predicted logits.
        target (torch.Tensor): Ground truth mask.

    Returns:
        float: Dice coefficient.
    """
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / (output.sum() + target.sum() + smooth)


def dice_coef2(output, target):
    """
    Computes Dice coefficient using thresholded output (for validation).

    Args:
        output (torch.Tensor): Thresholded prediction (0 or 1).
        target (torch.Tensor): Ground truth mask.

    Returns:
        float: Dice coefficient.
    """
    smooth = 1e-5

    output = output.view(-1)
    output = (output > 0.5).float().cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / (output.sum() + target.sum() + smooth)

# ------------------------
# Confusion Matrix Metrics
# ------------------------

def calculate_precision(tp, fp):
    """
    Calculates precision.

    Args:
        tp (int): True positives.
        fp (int): False positives.

    Returns:
        float: Precision score.
    """
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)


def calculate_recall(tp, fn):
    """
    Calculates recall.

    Args:
        tp (int): True positives.
        fn (int): False negatives.

    Returns:
        float: Recall score.
    """
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)


def calculate_fpps(fp, total_slices):
    """
    Calculates false positives per scan (FPPS).

    Args:
        fp (int): Number of false positives.
        total_slices (int): Total number of scans.

    Returns:
        float: FPPS score.
    """
    if total_slices == 0:
        return 0.0
    return fp / total_slices


def calculate_accuracy(tp, tn, fp, fn):
    """
    Calculates overall accuracy.

    Args:
        tp (int): True positives.
        tn (int): True negatives.
        fp (int): False positives.
        fn (int): False negatives.

    Returns:
        float: Accuracy score.
    """
    total = tp + tn + fp + fn
    if total == 0:
        return 0.0
    return (tp + tn) / total


# ------------------------
# File Saving Utility
# ------------------------

def save_metrics_to_csv(metrics, output_dir, filename="metrics.csv"):
    """
    Saves metrics dictionary to a CSV file.

    Args:
        metrics (dict): Dictionary of metric names and values.
        output_dir (str): Directory to save the file.
        filename (str, optional): Output file name. Defaults to "metrics.csv".
    """
    import os
    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, filename)

    metrics_df = pd.DataFrame(metrics.items(), columns=["Metric", "Result"])
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")

# ------------------------
# Confusion Matrix Computations
# ------------------------

def calculate_fp(prediction_dir, mask_dir, distance_threshold=80):
    """
    Calculates confusion matrix (TP, TN, FP, FN) for the normal dataset.

    Args:
        prediction_dir (str): Directory containing predicted masks.
        mask_dir (str): Directory containing ground truth masks.
        distance_threshold (float): Maximum distance for matching nodules.

    Returns:
        np.ndarray: Confusion matrix [TP, TN, FP, FN].
    """
    confusion_matrix = [0, 0, 0, 0]  # TP, TN, FP, FN
    s = generate_binary_structure(2, 2)

    print('Length of prediction dir is', len(os.listdir(prediction_dir)))
    for prediction in os.listdir(prediction_dir):
        pid = 'LIDC-IDRI-' + prediction[:4]
        mask_id = prediction.replace('PD', 'MA')
        mask = np.load(os.path.join(mask_dir, pid, mask_id))
        predict = np.load(os.path.join(prediction_dir, prediction))

        answer_com = np.array(ndi.center_of_mass(mask))
        patience = 0
        labeled_array, nf = label(predict, structure=s)

        if nf > 0:
            for n in range(nf):
                lab = np.array(labeled_array)
                lab[lab != (n + 1)] = 0
                lab[lab == (n + 1)] = 1
                predict_com = np.array(ndi.center_of_mass(labeled_array))
                if np.linalg.norm(predict_com - answer_com, 2) < distance_threshold:
                    patience += 1
                else:
                    confusion_matrix[2] += 1  # FP

            if patience > 0:
                confusion_matrix[0] += 1  # TP
            else:
                confusion_matrix[3] += 1  # FN
        else:
            confusion_matrix[3] += 1  # FN if no predictions found

    return np.array(confusion_matrix)


def calculate_fp_clean_dataset(prediction_dir, distance_threshold=80):
    """
    Calculates confusion matrix (TP, TN, FP, FN) for the clean dataset (no true nodules).

    Args:
        prediction_dir (str): Directory containing predicted masks.
        distance_threshold (float): Maximum distance for matching false positives.

    Returns:
        np.ndarray: Confusion matrix [TP, TN, FP, FN].
    """
    confusion_matrix = [0, 0, 0, 0]  # TP, TN, FP, FN
    s = generate_binary_structure(2, 2)

    for prediction in os.listdir(prediction_dir):
        predict = np.load(os.path.join(prediction_dir, prediction))
        patience = 0
        labeled_array, nf = label(predict, structure=s)

        if nf > 0:
            previous_com = np.array([-1, -1])
            for n in range(nf):
                lab = np.array(labeled_array)
                lab[lab != (n + 1)] = 0
                lab[lab == (n + 1)] = 1
                predict_com = np.array(ndi.center_of_mass(labeled_array))

                if previous_com[0] == -1:
                    confusion_matrix[2] += 1  # FP
                    previous_com = predict_com
                    continue

                if np.linalg.norm(previous_com - predict_com, 2) > distance_threshold:
                    if patience != 0:
                        continue
                    confusion_matrix[2] += 1  # FP
                    patience += 1
        else:
            confusion_matrix[1] += 1  # TN if no predictions found

    return np.array(confusion_matrix)
