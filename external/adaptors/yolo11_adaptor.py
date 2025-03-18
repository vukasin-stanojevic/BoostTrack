import torch
import torch.nn as nn
from ultralytics import YOLO

class PostModel(nn.Module):
    def __init__(self, model, conf, iou):
        super().__init__()
        self.model = model
        self.conf = conf
        self.iou = iou

    def forward(self, batch):
        """
        Runs inference and returns:
        1. Nx5, (x1, y1, x2, y2, conf)  → Original Format
        """
        rgb_means = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to('cuda')
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to('cuda')

        denormalized_tensor = (batch * std) + rgb_means
        normalized_batch = denormalized_tensor / denormalized_tensor.amax(dim=(1, 2, 3), keepdim=True)
        results = self.model(normalized_batch, conf=self.conf, iou=self.iou, verbose=False)

        preds = []
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy
                box_conf = result.boxes.conf

                if boxes.numel() == 0:  # Skip if no boxes exist
                    continue
                
                # Combine into final formats
                final_boxes = torch.cat((boxes, box_conf.unsqueeze(1)), dim=1)

                preds.append(final_boxes)

        if len(preds) == 0:
            return None

        pred = preds[0]  # Take the first batch

        if pred.numel() == 0:  # Ensure it's not empty
            return None
        return pred


def get_model(conf=0.1, iou_thresh=0.7, weights_path="external/weights/yoloV11_best.pt"):
    """
    Loads the pretrained YOLOv11 model from Ultralytics and wraps it for inference.
    """
    model = YOLO(weights_path)
    # model.fuse()
    # model = model.half()
    model = PostModel(model=model, conf=conf, iou=iou_thresh)
    model.cuda()
    return model