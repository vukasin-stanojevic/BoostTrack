"""
    This script is adopted from the SORT script by Alex Bewley alex@bewley.ai
"""
from __future__ import print_function

import os
from copy import deepcopy
from typing import Optional, List

import cv2
import numpy as np

from default_settings import GeneralSettings, BoostTrackSettings, BoostTrackPlusPlusSettings
from tracker.embedding import EmbeddingComputer
from tracker.assoc import associate, iou_batch, MhDist_similarity, shape_similarity, soft_biou_batch
from tracker.ecc import ECC
from tracker.kalmanfilter import KalmanFilter

def convert_bbox_to_z(bbox):
    """
    바운딩 박스를 변환하여 Kalman Filter에서 사용하는 형식으로 변환

    Args:
        bbox (list or np.ndarray): 바운딩 박스 좌표 [x1, y1, x2, y2]

    Returns:
        np.ndarray: 변환된 좌표 [x, y, h, r] (중심점, 높이, 종횡비)
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0

    r = w / float(h + 1e-6)

    return np.array([x, y, h, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Kalman Filter 상태 벡터를 일반적인 바운딩 박스 형식으로 변환

    Args:
        x (np.ndarray): 상태 벡터 [x, y, h, r]
        score (float, optional): 탐지 신뢰도 점수

    Returns:
        np.ndarray: 변환된 바운딩 박스 [x1, y1, x2, y2] 또는 [x1, y1, x2, y2, score]
    """

    h = x[2]
    r = x[3]
    w = 0 if r <= 0 else r * h

    if score is None:
        return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]).reshape((1, 5))


class KalmanBoxTracker(object):
    """
    개별 객체의 상태를 추적하는 Kalman 필터 기반 추적기 클래스
    """

    count = 0

    def __init__(self, bbox, emb: Optional[np.ndarray] = None):
        """
        Kalman 필터를 사용하여 객체를 추적하는 클래스 초기화

        Args:
            bbox (np.ndarray): 초기 바운딩 박스
            emb (np.ndarray, optional): 객체의 임베딩 벡터
        """

        self.bbox_to_z_func = convert_bbox_to_z
        self.x_to_bbox_func = convert_x_to_bbox

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        self.kf = KalmanFilter(self.bbox_to_z_func(bbox))
        self.emb = emb
        self.hit_streak = 0
        self.age = 0

    def get_confidence(self, coef: float = 0.9) -> float:
        """
        객체 추적 신뢰도를 계산
        """
        n = 7

        if self.age < n:
            return coef ** (n - self.age)
        return coef ** (self.time_since_update-1)

    def update(self, bbox: np.ndarray, score: float = 0):
        """
        Kalman 필터 상태를 업데이트하는 메서드
        """

        self.time_since_update = 0
        self.hit_streak += 1
        self.kf.update(self.bbox_to_z_func(bbox), score)

    def camera_update(self, transform: np.ndarray):
        x1, y1, x2, y2 = self.get_state()[0]
        x1_, y1_, _ = transform @ np.array([x1, y1, 1]).T
        x2_, y2_, _ = transform @ np.array([x2, y2, 1]).T
        w, h = x2_ - x1_, y2_ - y1_
        cx, cy = x1_ + w / 2, y1_ + h / 2
        self.kf.x[:4] = [cx, cy, h,  w / h]

    def predict(self):
        """
        Kalman 필터를 이용하여 객체의 다음 위치를 예측
        """

        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        return self.get_state()

    def get_state(self):
        """
        현재 객체의 바운딩 박스를 반환
        """
        return self.x_to_bbox_func(self.kf.x)

    def update_emb(self, emb, alpha=0.9):
        self.emb = alpha * self.emb + (1 - alpha) * emb
        self.emb /= np.linalg.norm(self.emb)

    def get_emb(self):
        return self.emb


class BoostTrack(object):
    """
    BoostTrack 객체 추적 클래스
    """
    def __init__(self, video_name: Optional[str] = None):
        """
        BoostTrack 객체를 초기화

        Args:
            video_name (str, optional): 추적할 비디오 파일 이름
        """

        self.frame_count = 0 # 처리된 프레임 수
        self.trackers: List[KalmanBoxTracker] = [] # 활성화된 추적기 리스트

        self.max_age = GeneralSettings.max_age(video_name) # 최대 추적 유지 시간
        self.iou_threshold = GeneralSettings['iou_threshold'] # IOU 임계값
        self.det_thresh = GeneralSettings['det_thresh'] # 탐지 신뢰도 임계값
        self.min_hits = GeneralSettings['min_hits'] # 트래커가 활성화되기 위한 최소 감지 횟수

        self.lambda_iou = BoostTrackSettings['lambda_iou']
        self.lambda_mhd = BoostTrackSettings['lambda_mhd']
        self.lambda_shape = BoostTrackSettings['lambda_shape']
        self.use_dlo_boost = BoostTrackSettings['use_dlo_boost']
        self.use_duo_boost = BoostTrackSettings['use_duo_boost']
        self.dlo_boost_coef = BoostTrackSettings['dlo_boost_coef']

        self.use_rich_s = BoostTrackPlusPlusSettings['use_rich_s']
        self.use_sb = BoostTrackPlusPlusSettings['use_sb']
        self.use_vt = BoostTrackPlusPlusSettings['use_vt']

        # 임베딩 기능 활성화 여부
        if GeneralSettings['use_embedding']:
            self.embedder = EmbeddingComputer(GeneralSettings['dataset'], GeneralSettings['test_dataset'], True)
        else:
            self.embedder = None

        # 카메라 모션 보정 기능 활성화 여부
        if GeneralSettings['use_ecc']:
            self.ecc = ECC(scale=350, video_name=video_name, use_cache=True)
        else:
            self.ecc = None

    def update(self, dets, img_tensor, img_numpy, tag):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.

        객체 탐지를 바탕으로 추적기 업데이트

        Args:
            dets (numpy.ndarray): [[x1, y1, x2, y2, score], ...] 형태의 탐지 결과
            img_tensor (Tensor): 현재 프레임의 PyTorch 텐서
            img_numpy (numpy.ndarray): 현재 프레임의 numpy 배열
            tag (str): 현재 프레임을 식별하는 태그

        Returns:
            numpy.ndarray: [[x1, y1, x2, y2, ID], ...] 형태의 추적 결과
        """
        if dets is None:
            return np.empty((0, 5))
        if not isinstance(dets, np.ndarray):
            dets = dets.cpu().detach().numpy()

        self.frame_count += 1 # 프레임 카운트 증가

        # Rescale
        # 크기 조정 (탐지 결과를 이미지 크기에 맞게 변환)
        scale = min(img_tensor.shape[2] / img_numpy.shape[0], img_tensor.shape[3] / img_numpy.shape[1])
        dets = deepcopy(dets)
        dets[:, :4] /= scale

        if self.ecc is not None:
            transform = self.ecc(img_numpy, self.frame_count, tag)
            for trk in self.trackers:
                trk.camera_update(transform)

        # get predicted locations from existing trackers.
        # 기존 트래커의 예상 위치 저장
        trks = np.zeros((len(self.trackers), 5))
        confs = np.zeros((len(self.trackers), 1))

        for t in range(len(trks)):
            pos = self.trackers[t].predict()[0]
            confs[t] = self.trackers[t].get_confidence()
            trks[t] = [pos[0], pos[1], pos[2], pos[3], confs[t, 0]]

        # BoostTrack++의 신뢰도 부스팅 적용
        if self.use_dlo_boost:
            dets = self.dlo_confidence_boost(dets, self.use_rich_s, self.use_sb, self.use_vt) # DLO(Dynamic Likelihood Optimization): 탐지 신뢰도 조정
            # use_rich_s, use_sb, use_vt는 BoostTrack++에서 DLO 부스팅을 조정하는 파라미터

        if self.use_duo_boost:
            dets = self.duo_confidence_boost(dets) # DUO(Detection Uncertainty Optimization): 신뢰도가 낮은 탐지 결과를 보정

        remain_inds = dets[:, 4] >= self.det_thresh # 신뢰도가 det_thresh보다 높은 탐지 결과만 유지
        dets = dets[remain_inds]
        scores = dets[:, 4]

        # Generate embeddings
        # 탐지된 객체와 기존 트래커 간의 임베딩 기반 매칭 수행
        # compute_embedding()을 이용하여 탐지된 객체의 특징 벡터를 계산
        dets_embs = np.ones((dets.shape[0], 1))
        emb_cost = None # 트래커의 임베딩을 추출하여 코사인 유사도를 기반으로 emb_cost 행렬을 생성
        if self.embedder and dets.size > 0:
            dets_embs = self.embedder.compute_embedding(img_numpy, dets[:, :4], tag)
            trk_embs = []
            for t in range(len(self.trackers)):
                trk_embs.append(self.trackers[t].get_emb())
            trk_embs = np.array(trk_embs)
            if trk_embs.size > 0 and dets.size > 0:
                emb_cost = dets_embs.reshape(dets_embs.shape[0], -1) @ trk_embs.reshape((trk_embs.shape[0], -1)).T
        emb_cost = None if self.embedder is None else emb_cost

        # 탐지된 객체(dets)와 기존 트래커(trks) 간의 매칭 수행
        # associate() 함수는 매칭된 객체(matched), 매칭되지 않은 탐지(unmatched_dets), 매칭되지 않은 트래커(unmatched_trks) 반환
        matched, unmatched_dets, unmatched_trks, sym_matrix = associate(
            dets,
            trks,
            self.iou_threshold,
            mahalanobis_distance=self.get_mh_dist_matrix(dets),
            track_confidence=confs,
            detection_confidence=scores,
            emb_cost=emb_cost,
            lambda_iou=self.lambda_iou,
            lambda_mhd=self.lambda_mhd,
            lambda_shape=self.lambda_shape
        )

        # 탐지 신뢰도가 높을수록 dets_alpha 값을 증가시켜 탐지 신뢰도가 높은 객체를 더 반영하도록 설정
        trust = (dets[:, 4] - self.det_thresh) / (1 - self.det_thresh)
        af = 0.95
        dets_alpha = af + (1 - af) * (1 - trust)

        # 매칭된 객체는 기존 트래커를 업데이트하고, update_emb()를 통해 임베딩도 갱신
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :], scores[m[0]])
            self.trackers[m[1]].update_emb(dets_embs[m[0]], alpha=dets_alpha[m[0]])

        # 매칭되지 않은 탐지 결과 중 신뢰도가 높은 경우 새로운 Kalman 필터 기반 트래커를 생성
        for i in unmatched_dets:
            if dets[i, 4] >= self.det_thresh:
                self.trackers.append(KalmanBoxTracker(dets[i, :], emb=dets_embs[i]))

        # trk.time_since_update > self.max_age: 일정 시간 이상 갱신되지 않은 객체 삭제
        # trk.hit_streak >= self.min_hits: 일정 횟수 이상 감지된 객체만 유지
        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # +1 as MOT benchmark requires positive
                ret.append(np.concatenate((d, [trk.id + 1], [trk.get_confidence()])).reshape(1, -1))
            i -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

    def dump_cache(self):
        if self.ecc is not None:
            self.ecc.save_cache()

    def get_iou_matrix(self, detections: np.ndarray, buffered: bool = False) -> np.ndarray: # 탐지된 객체(detections)와 기존 트래커(trackers) 간의 IoU 행렬을 계산
        trackers = np.zeros((len(self.trackers), 5))
        for t, trk in enumerate(trackers):
            pos = self.trackers[t].get_state()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], self.trackers[t].get_confidence()]

        return iou_batch(detections, trackers) if not buffered else soft_biou_batch(detections, trackers)

    def get_mh_dist_matrix(self, detections: np.ndarray, n_dims: int = 4) -> np.ndarray: # Mahalanobis 거리 기반 매칭을 위해 탐지된 객체와 트래커 간의 거리 행렬을 계산
        if len(self.trackers) == 0:
            return np.zeros((0, 0))
        z = np.zeros((len(detections), n_dims), dtype=float)
        x = np.zeros((len(self.trackers), n_dims), dtype=float)
        sigma_inv = np.zeros_like(x, dtype=float)

        f = self.trackers[0].bbox_to_z_func
        for i in range(len(detections)):
            z[i, :n_dims] = f(detections[i, :]).reshape((-1, ))[:n_dims]
        for i in range(len(self.trackers)):
            x[i] = self.trackers[i].kf.x[:n_dims]
            # Note: we assume diagonal covariance matrix
            sigma_inv[i] = np.reciprocal(np.diag(self.trackers[i].kf.covariance[:n_dims, :n_dims]))

        return ((z.reshape((-1, 1, n_dims)) - x.reshape((1, -1, n_dims))) ** 2 * sigma_inv.reshape((1, -1, n_dims))).sum(axis=2)

    def duo_confidence_boost(self, detections: np.ndarray) -> np.ndarray:  # DUO 부스팅: Mahalanobis 거리 기반 탐지 신뢰도 조정
        n_dims = 4
        limit = 13.2767
        mahalanobis_distance = self.get_mh_dist_matrix(detections, n_dims)

        if mahalanobis_distance.size > 0 and self.frame_count > 1:
            min_mh_dists = mahalanobis_distance.min(1)

            mask = (min_mh_dists > limit) & (detections[:, 4] < self.det_thresh)
            boost_detections = detections[mask]
            boost_detections_args = np.argwhere(mask).reshape((-1,))
            iou_limit = 0.3
            if len(boost_detections) > 0:
                bdiou = iou_batch(boost_detections, boost_detections) - np.eye(len(boost_detections))
                bdiou_max = bdiou.max(axis=1)

                remaining_boxes = boost_detections_args[bdiou_max <= iou_limit]
                args = np.argwhere(bdiou_max > iou_limit).reshape((-1,))
                for i in range(len(args)):
                    boxi = args[i]
                    tmp = np.argwhere(bdiou[boxi] > iou_limit).reshape((-1,))
                    args_tmp = np.append(np.intersect1d(boost_detections_args[args], boost_detections_args[tmp]), boost_detections_args[boxi])

                    conf_max = np.max(detections[args_tmp, 4])
                    if detections[boost_detections_args[boxi], 4] == conf_max:
                        remaining_boxes = np.array(remaining_boxes.tolist() + [boost_detections_args[boxi]])

                mask = np.zeros_like(detections[:, 4], dtype=np.bool_)
                mask[remaining_boxes] = True

            detections[:, 4] = np.where(mask, self.det_thresh + 1e-4, detections[:, 4])

        return detections

    def dlo_confidence_boost(self, detections: np.ndarray, use_rich_sim: bool, use_soft_boost: bool, use_varying_th: bool) -> np.ndarray: # DLO 부스팅: IoU, Mahalanobis 거리 및 형태 유사성을 활용한 신뢰도 조정
        sbiou_matrix = self.get_iou_matrix(detections, True)
        if sbiou_matrix.size == 0:
            return detections
        trackers = np.zeros((len(self.trackers), 6))
        for t, trk in enumerate(trackers):
            pos = self.trackers[t].get_state()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0, self.trackers[t].time_since_update - 1]

        if use_rich_sim:
            mhd_sim = MhDist_similarity(self.get_mh_dist_matrix(detections), 1)
            shape_sim = shape_similarity(detections, trackers)
            S = (mhd_sim + shape_sim + sbiou_matrix) / 3
        else:
            S = self.get_iou_matrix(detections, False)

        if not use_soft_boost and not use_varying_th:
            max_s = S.max(1)
            coef = self.dlo_boost_coef
            detections[:, 4] = np.maximum(detections[:, 4], max_s * coef)

        else:
            if use_soft_boost:
                max_s = S.max(1)
                alpha = 0.65
                detections[:, 4] = np.maximum(detections[:, 4], alpha*detections[:, 4] + (1-alpha)*max_s**(1.5))
            if use_varying_th:
                threshold_s = 0.95
                threshold_e = 0.8
                n_steps = 20
                alpha = (threshold_s - threshold_e) / n_steps
                tmp = (S > np.maximum(threshold_s - trackers[:, 5] * alpha, threshold_e)).max(1)
                scores = deepcopy(detections[:, 4])
                scores[tmp] = np.maximum(scores[tmp], self.det_thresh + 1e-5)

                detections[:, 4] = scores

        return detections
