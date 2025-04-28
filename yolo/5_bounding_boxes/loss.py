import torch
import torch.nn as nn
from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=5, C=20):
        super(YoloLoss,self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B*5)
        
        ious = []
        for i in range(self.B):
            iou = intersection_over_union(
                predictions[..., (21 + (i * 5)):(25 + (i * 5))], target[..., 21:25]
            )
            ious.append(iou.unsqueeze(0))

        ious = torch.cat(ious, dim=0)
        ious_maxes, best_box = torch.max(ious, dim=0)
        exists_box = target[..., 20].unsqueeze(3) #identity_obj_i, is there an object in cell i?

        #BOX COORDINATES
        # box_predictions = exists_box * sum(
        #     [
        #         #Best Box will be either 0 or 1 depending on which box is used if there exists a box
        #         # (best_box == i).unsqueeze(-1) * predictions[..., 21 + i * 5:25 + i * 5]
        #         # Ensure mask shape is (N, S, S, 1)
        #         mask = (best_box == i).float().reshape(predictions.shape[:3] + (1,))

        #         # Get the corresponding box prediction
        #         box_pred = predictions[..., 21 + i * 5 : 25 + i * 5]  # shape (N, S, S, 4)

        #         # Apply mask
        #         masked_pred = mask * box_pred

        #         for i in range(self.B)
        #     ]
        # )
        box_predictions = (
            best_box.eq(0).float() * predictions[..., 21:25] +
            best_box.eq(1).float() * predictions[..., 26:30] +
            best_box.eq(2).float() * predictions[..., 31:35] +
            best_box.eq(3).float() * predictions[..., 36:40] +
            best_box.eq(4).float() * predictions[..., 41:45]
        )
        
        box_targets = target[..., 21:25]
        
        # box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4] * torch.sqrt(
        #     torch.abs(box_predictions[..., 2:4] + 1e-6) #If pred is negative or = 0
        #     )
        # )

        # replace in-place sqrt with out-of-place ops
        box_predictions_xy = box_predictions[..., :2]
        box_predictions_wh = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4]) + 1e-6
        )
        box_predictions = torch.cat([box_predictions_xy, box_predictions_wh], dim=-1)

        box_targets_xy = box_targets[..., :2]
        box_targets_wh = torch.sqrt(box_targets[..., 2:4] + 1e-6)
        box_targets = torch.cat([box_targets_xy, box_targets_wh], dim=-1)

        box_loss = self.mse(
            torch.flatten(exists_box * box_predictions, end_dim=-2),
            torch.flatten(exists_box * box_targets, end_dim=-2),
        )

        # box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        # # (N, S, S, 4) -> (N*S*S, 4)
        # box_loss = self.mse(
        #     torch.flatten(exists_box * box_predictions, end_dim=-2),
        #     torch.flatten(exists_box * box_targets, end_dim=-2),
        # )

        #OBJECT LOSS
        #Which box is responsible for the object
        # pred_box = sum(
        #     (best_box == i).unsqueeze(-1) * predictions[..., 20 + i * 5:21 + i * 5]
        #     for i in range(self.B)
        # )
        pred_box = (
            best_box.eq(0).float() * predictions[..., 20:21] +
            best_box.eq(1).float() * predictions[..., 25:26] +
            best_box.eq(2).float() * predictions[..., 30:31] +
            best_box.eq(3).float() * predictions[..., 35:36] + 
            best_box.eq(4).float() * predictions[..., 40:41]
        )

        # (N*S*S) Does there actually exist a box?
        # iou_scores = ious_maxes.unsqueeze(-1)
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box, end_dim=-2),
            torch.flatten(exists_box * target[..., 20:21], end_dim=-2), 
        )

        #NO OBJECT LOSS
        #(N, S, S, 1) -> (N, S*S)
        no_object_loss = 0
        for i in range(self.B):
            no_object_loss += self.mse(
                torch.flatten((1 - exists_box) * predictions[..., 20 + i * 5:21 + i * 5], start_dim=1),
                torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
            )

        # no_object_loss += self.mse(
        #     torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
        #     torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        # )

        #CLASS LOSS

        #(N, S, S, C) -> (N*S*S, C)
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2),
            torch.flatten(exists_box * target[..., :20], end_dim=-2),
        )

        loss = (
            self.lambda_coord * box_loss # First 2 rows of loss function in paper
            + object_loss # 3rd row of loss function in paper
            + self.lambda_noobj * no_object_loss # 4th row of loss function in paper
            + class_loss # Last row of loss function in paper
        )

        return loss