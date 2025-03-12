"""Generic detector."""
import os
import pickle

import torch

from external.adaptors import yolox_adaptor


class Detector(torch.nn.Module): # 객체탐지를 위한 Detector클래스, 미리 정의된 모델은 yolox밖에 없음
    K_MODELS = {"yolox"}

    def __init__(self, model_type, path, dataset):
        super().__init__()
        if model_type not in self.K_MODELS:
            raise RuntimeError(f"{model_type} detector not supported")

        self.model_type = model_type # 사용할 모델의 타입
        self.path = path # 모델 가중치 파일 경로
        self.dataset = dataset # 사용할 데이터셋 종류
        self.model = None

        os.makedirs("./cache", exist_ok=True) # 캐시 저장 폴더 생성
        self.cache_path = os.path.join(
            "./cache", f"det_{os.path.basename(path).split('.')[0]}.pkl"
        )
        self.cache = {}
        if os.path.exists(self.cache_path): # 이전에 저장된 탐지 결과 캐시 로드

            with open(self.cache_path, "rb") as fp:
                self.cache = pickle.load(fp)
        else:
            self.initialize_model()

    def initialize_model(self):
        """
        YOLOX 모델을 초기화하는 메서드.

        - 모델 타입이 'yolox'인 경우, YOLOX 어댑터를 통해 모델을 불러옴.
        """
        if self.model_type == "yolox":
            self.model = yolox_adaptor.get_model(self.path, self.dataset) # 모델 타입이 'yolox'인 경우, YOLOX 어댑터를 통해 모델을 불러옴

    def detect(self, img):
        """
        입력 이미지를 사용하여 객체 탐지를 수행하는 메서드

        Args:
            img (torch.Tensor): 입력 이미지 텐서

        Returns:
            torch.Tensor: 탐지된 객체의 출력 결과
        """
        if self.model is None:
            self.initialize_model()
        
        with torch.no_grad():
            img = img.half()  # 입력을 `float16`으로 변환하여 연산 최적화
            output = self.model(img)
        return output

    def forward(self, batch, tag=None):
        """
        배치 데이터를 입력받아 모델의 forward 연산을 수행하는 메서드

        Args:
            batch (torch.Tensor): 입력 이미지 배치 텐서
            tag (str, optional): 탐지 결과를 캐시할 키

        Returns:
            torch.Tensor: 탐지 결과
        """
        if tag in self.cache:
            return self.cache[tag]
        if self.model is None:
            self.initialize_model()

        with torch.no_grad():
            batch = batch.half()
            output = self.model(batch)
        if output is not None:
            self.cache[tag] = output.cpu().detach()

        return output

    def dump_cache(self):
        """
        탐지 결과 캐시를 파일로 저장하는 메서드.

        - 탐지 결과를 pickle을 사용하여 캐시 파일로 저장함
        """
        with open(self.cache_path, "wb") as fp:
            pickle.dump(self.cache, fp)
