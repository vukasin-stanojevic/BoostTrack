import warnings
from typing import Optional

import lap
import numpy as np
from copy import deepcopy
import scipy.spatial as sp


def split_cosine_dist(dets, trks, affinity_thresh=0.55, pair_diff_thresh=0.6, hard_thresh=True):

    cos_dist = np.zeros((len(dets), len(trks)))

    for i in range(len(dets)):
        for j in range(len(trks)):

            cos_d = 1 - sp.distance.cdist(dets[i], trks[j], "cosine")  ## shape = 3x3
            patch_affinity = np.max(cos_d, axis=0)  ## shape = [3,]
            # exp16 - Using Hard threshold
            if hard_thresh:
                if len(np.where(patch_affinity > affinity_thresh)[0]) != len(patch_affinity):
                    cos_dist[i, j] = 0
                else:
                    cos_dist[i, j] = np.max(patch_affinity)
            else:
                cos_dist[i, j] = np.max(patch_affinity)  # can experiment with mean too (max works slightly better)

    return cos_dist

def shape_similarity(detects: np.ndarray, tracks: np.ndarray) -> np.ndarray:
    if detects.size == 0 or tracks.size == 0:
        return np.zeros((0, 0))

    dw = (detects[:, 2] - detects[:, 0]).reshape((-1, 1))
    dh = (detects[:, 3] - detects[:, 1]).reshape((-1, 1))
    tw = (tracks[:, 2] - tracks[:, 0]).reshape((1, -1))
    th = (tracks[:, 3] - tracks[:, 1]).reshape((1, -1))
    return np.exp(-(np.abs(dw - tw)/np.maximum(dw, tw) + np.abs(dh - th)/np.maximum(dw, tw)))

def iou_batch(bboxes1, bboxes2):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    o = wh / (
        (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
        - wh
    )

    return o



def linear_assignment(detections: np.ndarray, trackers: np.ndarray,
                      iou_matrix: np.ndarray, cost_matrix: np.ndarray,
                      threshold: float, emb_cost: Optional[np.ndarray] = None):
    if iou_matrix is None and cost_matrix is None:
        raise Exception("Both iou_matrix and cost_matrix are None!")
    if iou_matrix is None:
        iou_matrix = deepcopy(cost_matrix)
    if cost_matrix is None:
        cost_matrix = deepcopy(iou_matrix)
    if cost_matrix.size > 0:
        a = (cost_matrix > threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            _, x, y = lap.lapjv(-cost_matrix, extend_cost=True)
            matched_indices = np.array([[y[i], i] for i in x if i >= 0])
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        valid_match = iou_matrix[m[0], m[1]] >= threshold  or (False if emb_cost is None else (iou_matrix[m[0], m[1]] >= threshold / 2 and emb_cost[m[0], m[1]] >= 0.75))
        if valid_match:
            matches.append(m.reshape(1, 2))
        else:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


def associate(
        detections,
        trackers,
        iou_threshold,
        mahalanobis_distance: Optional[np.ndarray] = None,
        track_confidence: Optional[np.ndarray] = None,
        detection_confidence: Optional[np.ndarray] = None,
        emb_cost: Optional[np.ndarray] = None,
        lambda_iou: float = 0.5,
        lambda_mhd: float = 0.25,
        lambda_shape: float = 0.25
):
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0, 5), dtype=int),
        )
    iou_matrix = iou_batch(detections, trackers)

    cost_matrix = deepcopy(iou_matrix)

    if detection_confidence is not None and track_confidence is not None:
        conf = np.multiply(detection_confidence.reshape((-1, 1)), track_confidence.reshape((1, -1)))

        conf[iou_matrix < iou_threshold] = 0

        cost_matrix += lambda_iou * conf * iou_matrix
    else:
        warnings.warn("Detections or tracklet confidence is None and detection-tracklet confidence cannot be computed!")
        conf = None

    if mahalanobis_distance is not None and mahalanobis_distance.size > 0:
        limit = 13.2767  # 99% conf interval https://www.mathworks.com/help/stats/chi2inv.html

        mask = mahalanobis_distance > limit
        mahalanobis_distance[mask] = limit
        mahalanobis_distance = limit - mahalanobis_distance

        mahalanobis_distance = np.exp(mahalanobis_distance) / np.exp(mahalanobis_distance).sum(0).reshape((1, -1))
        mahalanobis_distance = np.where(mask, 0, mahalanobis_distance)

        cost_matrix += lambda_mhd * mahalanobis_distance
        if conf is not None:
            cost_matrix += lambda_shape * conf * shape_similarity(detections, trackers)

    if emb_cost is not None:
        lambda_emb = 3
        cost_matrix += lambda_emb * emb_cost

    return linear_assignment(detections, trackers, iou_matrix, cost_matrix, iou_threshold, emb_cost)