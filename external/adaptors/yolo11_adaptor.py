import torch
import torch.nn as nn
from ultralytics import YOLO
 
class PostModel(nn.Module):
     def __init__(self, model, conf):
         super().__init__()
         self.model = model
         self.conf = conf
 
     def forward(self, batch):
         """
         Runs inference and returns:
         1. Nx5, (x1, y1, x2, y2, conf)  → Original Format
         """
         rgb_means = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to('cuda')
         std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to('cuda')
 
         denormalized_tensor = (batch * std) + rgb_means
 
         normalized_batch = denormalized_tensor / denormalized_tensor.amax(dim=(1, 2, 3), keepdim=True)
 
         results = self.model(normalized_batch, conf=self.conf)
         
         preds_xywh = []  # Store (x_top_left, y_top_left, width, height, conf)
 
         for result in results:
             if result.boxes is not None:
                 boxes = result.boxes.xyxy  # Extract bounding boxes (x1, y1, x2, y2)
                 box_conf = result.boxes.conf  # Extract confidence scores
                 
                 # x1, y1, x2, y2 = result.boxes.xyxy[:, 0], result.boxes.xyxy[:, 1], result.boxes.xyxy[:, 2], result.boxes.xyxy[:, 3]
 
                 # Compute width and height
                 # width = x2 - x1
                 # height = y2 - y1
 
                 # Stack (x1, y1, width, height) together
                 # boxes_xywh = torch.stack((x1, y1, width, height), dim=1)
                 if boxes.numel() == 0:  # Skip if no boxes exist
                     continue
                 
                 # Combine into final formats
                 final_boxes = torch.cat((boxes, box_conf.unsqueeze(1)), dim=1)
 
                 preds_xywh.append(final_boxes)  # Append converted format
 
         if len(preds_xywh) == 0:
             return None
 
         pred = preds_xywh[0]  # Take the first batch
 
         if pred.numel() == 0:  # Ensure it's not empty
             return None
         
         return pred
 
def get_model(conf, weights_path="yolov11.pt"):
     """
     Loads the pretrained YOLOv11 model from Ultralytics and wraps it for inference.
     """
     model = YOLO(weights_path)
     # model.fuse()
     # model = model.half()
     model = PostModel(model, conf)
    #  model.cuda()
     return model