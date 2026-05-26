import numpy as np
from sklearn.metrics import jaccard_score

def pixel_accuracy(eval_segm, gt_segm):
    """
    sum_i(n_ii) / sum_i(t_i)
    """
    eval_segm = np.asarray(eval_segm)
    gt_segm = np.asarray(gt_segm)
    
    cl = np.unique(gt_segm)
    n_cl = len(cl)
    
    sum_n_ii = 0
    sum_t_i  = 0

    for c in cl:
        curr_eval_mask = (eval_segm == c)
        curr_gt_mask = (gt_segm == c)

        sum_n_ii += np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        sum_t_i  += np.sum(curr_gt_mask)
 
    if sum_t_i == 0:
        return 0.0
    return float(sum_n_ii) / sum_t_i

def mean_IU(eval_segm, gt_segm):
    """
    Returns the IoU for positive class (class 1)
    """
    eval_segm = np.asarray(eval_segm).ravel()
    gt_segm = np.asarray(gt_segm).ravel()
    scores = jaccard_score(gt_segm, eval_segm, average=None)
    if len(scores) > 1:
        return scores[1]
    # If only one class present
    return scores[0] if gt_segm[0] == 1 else 0.0

def macro_iou(eval_segm, gt_segm):
    """
    Returns the macro average IoU over all classes (class 0 and class 1)
    """
    eval_segm = np.asarray(eval_segm).ravel()
    gt_segm = np.asarray(gt_segm).ravel()
    return jaccard_score(gt_segm, eval_segm, average='macro')

def frequency_weighted_IU(eval_segm, gt_segm):
    """
    sum_k(t_k)^(-1) * sum_i((t_i*n_ii)/(t_i + sum_j(n_ji) - n_ii))
    """
    eval_segm = np.asarray(eval_segm)
    gt_segm = np.asarray(gt_segm)
    
    eval_cl = np.unique(eval_segm)
    gt_cl = np.unique(gt_segm)
    cl = np.union1d(eval_cl, gt_cl)
    n_cl = len(cl)

    frequency_weighted_IU_ = [0.0] * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = (eval_segm == c)
        curr_gt_mask = (gt_segm == c)
 
        if np.sum(curr_eval_mask) == 0 or np.sum(curr_gt_mask) == 0:
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i  = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        frequency_weighted_IU_[i] = float(t_i * n_ii) / (t_i + n_ij - n_ii)
 
    sum_k_t_k = eval_segm.shape[0] * eval_segm.shape[1]
    
    return np.sum(frequency_weighted_IU_) / sum_k_t_k
