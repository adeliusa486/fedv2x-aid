"""Local objective of FedV2X-Aid on cached trunk features (Eq. (local) of the paper).

Box regression (CIoU) and distribution focal loss are Ultralytics 8.4's (task-aligned assignment, gains box 7.5,
cls 0.5, dfl 1.5). The classification term is *source-scoped*:

* A+  = anchors that the task-aligned assigner gives to a labelled box;
* A-  = anchors whose point lies neither inside a detector proposal with score >= tau_low (0.05) nor inside a
        range-sensor region;
* for a in A+ u A- and every class c in the attested set C_s of the frame's label sources, BCE(p_ac, y_ac)
  with y from the assigner (A+) or 0 (A-);
* every other anchor-class pair is distilled towards the received head: beta * BCE(p_ac, p_ac^(t)), beta = 0.5;
* each range region R adds lambda_T * L_set(R), L_set(R) = -mean_{a in A_R} log(1 - prod_{c in S_R}(1 - p_ac)),
  lambda_T = 0.5 (Eq. (setloss)).

With ``scoped=False`` the loss is the standard Ultralytics detection loss (every unlabelled anchor-class pair is
background): the self-training baselines and the "no source-scoped masks" ablation use it.
Box labels carry weights q: the assigner's target scores of the anchors of a box are multiplied by its q.

Several clients can be trained in one pass (src/fl/stacked.py): ``client`` gives the client of every image, and
every term is normalised per client exactly as Ultralytics normalises per batch (by the client's sum of target
scores, the set term by the client's number of images). The returned total is the sum of the clients' losses,
so the gradient with respect to each client's (separate) parameters is the gradient of that client's own loss.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from ultralytics.utils.loss import DFLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import TaskAlignedAssigner, bbox2dist, dist2bbox, make_anchors

BOX_GAIN, CLS_GAIN, DFL_GAIN = 7.5, 0.5, 1.5


class ScopedDetectionLoss:
    def __init__(self, nc: int, stride, reg_max=16, device="cuda", beta=0.5, lambda_t=0.5):
        self.nc, self.reg_max = nc, reg_max
        self.stride = torch.as_tensor(stride, dtype=torch.float32, device=device)
        self.device = device
        self.beta, self.lambda_t = beta, lambda_t
        self.assigner = TaskAlignedAssigner(topk=10, num_classes=nc, alpha=0.5, beta=6.0,
                                            stride=self.stride.tolist())
        self.dfl = DFLoss(reg_max)
        self.proj = torch.arange(reg_max, dtype=torch.float, device=device)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def decode(self, anchor_points, pred_dist):
        b, a, c = pred_dist.shape
        pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds: dict, batch: dict, teacher_logits=None, scoped=True, client=None, n_clients=1):
        """preds: raw head outputs {"boxes" (N,64,A), "scores" (N,nc,A), "feats" (list giving the grid shapes)}.

        batch (on the device; boxes in letterbox pixels, xyxy):
          gt_boxes (N, M, 4), gt_cls (N, M) long, gt_w (N, M), gt_mask (N, M) bool
          attested (N, nc) bool; prop_boxes (N, P, 4), prop_mask (N, P);
          reg_boxes (N, R, 4), reg_sets (N, R, nc) bool, reg_mask (N, R)          (scoped only)
        teacher_logits: (N, nc, A) class logits of the received head(s) (scoped only).
        client: (N,) long, client index of each image (default: all images belong to client 0).
        """
        pred_distri = preds["boxes"].permute(0, 2, 1).contiguous().float()
        pred_scores = preds["scores"].permute(0, 2, 1).contiguous().float()      # (N, A, nc) logits
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        N = pred_scores.shape[0]
        if client is None:
            client = torch.zeros(N, dtype=torch.long, device=self.device)
        K = n_clients
        pred_bboxes = self.decode(anchor_points, pred_distri)
        gt_boxes, gt_cls, gt_mask = batch["gt_boxes"], batch["gt_cls"], batch["gt_mask"]
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_boxes.dtype),
            anchor_points * stride_tensor, gt_cls.unsqueeze(-1).float(), gt_boxes, gt_mask.unsqueeze(-1).float())
        if gt_boxes.shape[1]:
            w = torch.gather(batch["gt_w"], 1, target_gt_idx)
            target_scores = target_scores * torch.where(fg_mask, w, torch.ones_like(w)).unsqueeze(-1)
        ts_img = target_scores.sum((1, 2))                                     # (N,)
        ts_cli = torch.zeros(K, device=self.device).index_add_(0, client, ts_img).clamp(min=1)
        norm = ts_cli[client]                                                   # (N,) per-image normaliser
        set_loss = pred_scores.new_zeros(())
        if not scoped:
            bce = self.bce(pred_scores, target_scores)                          # (N, A, nc)
            cls_img = bce.sum((1, 2))
            dist_img = torch.zeros_like(cls_img)
            del bce
        else:
            pts = (anchor_points * stride_tensor).unsqueeze(0)                  # (1, A, 2)

            def inside(boxes, mask):
                if boxes.shape[1] == 0:
                    return torch.zeros(N, pts.shape[1], dtype=torch.bool, device=self.device)
                x, y = pts[..., 0].unsqueeze(-1), pts[..., 1].unsqueeze(-1)
                bx = boxes.unsqueeze(1)
                ins = (x >= bx[..., 0]) & (x <= bx[..., 2]) & (y >= bx[..., 1]) & (y <= bx[..., 3])
                return (ins & mask.unsqueeze(1)).any(-1)
            in_prop = inside(batch["prop_boxes"], batch["prop_mask"])
            in_reg = inside(batch["reg_boxes"], batch["reg_mask"])
            a_neg = ~in_prop & ~in_reg & ~fg_mask
            scope = (fg_mask | a_neg).unsqueeze(-1) & batch["attested"].unsqueeze(1)   # (N, A, nc)
            # one BCE with combined targets: assigner targets inside the scope, the received head's
            # probabilities outside it (identical to two separate BCE terms, half the memory)
            if teacher_logits is not None and self.beta > 0:
                tgt = torch.where(scope, target_scores, teacher_logits.permute(0, 2, 1).float().sigmoid())
                bce = self.bce(pred_scores, tgt)
                del tgt
                cls_img = (bce * scope).sum((1, 2))
                dist_img = self.beta * (bce * ~scope).sum((1, 2))
            else:
                bce = self.bce(pred_scores, target_scores)
                cls_img = (bce * scope).sum((1, 2))
                dist_img = torch.zeros_like(cls_img)
            del bce, scope
            if batch["reg_boxes"].shape[1] and bool(batch["reg_mask"].any()):
                rows = batch["reg_mask"].any(1).nonzero(as_tuple=True)[0]          # images with a region
                cls_any = batch["reg_sets"][rows].any(1).any(0).nonzero(as_tuple=True)[0]   # admissible classes
                p = pred_scores[rows][:, :, cls_any].sigmoid()                  # (n, A, c)
                x, y = pts[0, :, 0], pts[0, :, 1]
                rb = batch["reg_boxes"][rows]                                   # (n, R, 4)
                ins = ((x[None, None] >= rb[..., 0:1]) & (x[None, None] <= rb[..., 2:3]) &
                       (y[None, None] >= rb[..., 1:2]) & (y[None, None] <= rb[..., 3:4]))      # (n, R, A)
                ins &= batch["reg_mask"][rows].unsqueeze(-1)
                sets = batch["reg_sets"][rows][:, :, cls_any].float()           # (n, R, c)
                # log prod_{c in S}(1 - p_ac) = sum_c S_c * log(1 - p_ac)
                log_miss = torch.einsum("nac,nrc->nra", torch.log1p(-p.clamp(max=1 - 1e-6)), sets)
                term = -torch.log1p(-log_miss.exp().clamp(max=1 - 1e-6))       # (n, R, A)
                cnt = ins.sum(-1).clamp(min=1)
                per_region = (term * ins).sum(-1) / cnt                         # (n, R)
                set_img = (per_region * ins.any(-1)).sum(-1)                    # (n,)
                imgs_per_client = torch.zeros(K, device=self.device).index_add_(
                    0, client, torch.ones(N, device=self.device)).clamp(min=1)
                set_loss = self.lambda_t * (set_img / imgs_per_client[client[rows]]).sum()
        fg = fg_mask.nonzero(as_tuple=True)
        weight = target_scores[fg].sum(-1, keepdim=True)
        iou = bbox_iou(pred_bboxes[fg], target_bboxes[fg] / stride_tensor[fg[1]], xywh=False, CIoU=True)
        box_term = ((1.0 - iou) * weight).squeeze(-1)
        target_ltrb = bbox2dist(anchor_points, target_bboxes / stride_tensor, self.reg_max - 1)
        dfl_term = (self.dfl(pred_distri[fg].view(-1, self.reg_max), target_ltrb[fg]) * weight).squeeze(-1)
        box_img = torch.zeros(N, device=self.device).index_add_(0, fg[0], box_term)
        dfl_img = torch.zeros(N, device=self.device).index_add_(0, fg[0], dfl_term)
        box_loss = (box_img / norm).sum()
        dfl_loss = (dfl_img / norm).sum()
        cls_loss = (cls_img / norm).sum()
        distill = (dist_img / norm).sum()
        total = BOX_GAIN * box_loss + CLS_GAIN * (cls_loss + distill) + DFL_GAIN * dfl_loss + set_loss
        parts = {"box": box_loss.detach(), "cls": cls_loss.detach(), "distill": distill.detach(),
                 "dfl": dfl_loss.detach(), "set": set_loss.detach()}
        return total, parts
