import numpy as np
import torch
from scipy.ndimage import label, generate_binary_structure
import torch.nn.functional as F
import os
from scipy import ndimage as ndi

def iou_score(output, target):
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
    smooth = 1e-5

    # we need to use sigmoid because the output of Unet is logit.
    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)


def dice_coef2(output, target):
    "This metric is for validation purpose"
    smooth = 1e-5

    output = output.view(-1)
    output = (output > 0.5).float().cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)


def calculate_precision(tp, fp):
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)


def calculate_recall(tp, fn):
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)


def calculate_fpps(fp, total_patients):
    if total_patients == 0:
        return 0.0
    return fp / total_patients


def save_metrics_to_csv(metrics, output_dir, filename="metrics.csv"):
    """
    Save metrics to a CSV file in the specified output directory.

    Args:
        metrics (dict): Dictionary of metric names and their values.
        output_dir (str): Directory where the CSV file will be saved.
        filename (str): Name of the output CSV file.
    """
    import os
    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, filename)

    metrics_df = pd.DataFrame(metrics.items(), columns=["Metric", "Result"])
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")



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
