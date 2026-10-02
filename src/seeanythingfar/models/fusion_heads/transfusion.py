import itertools
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from seeanythingfar.models.base import BaseFusionHead
from seeanythingfar.models.fusion_heads.assigner import HungarianAssigner3D
from seeanythingfar.models.losses import clip_sigmoid, gaussian_focal_loss, sigmoid_focal_loss
from seeanythingfar.utils.box_ops import CODE_SIZE, decode_boxes, draw_gaussian, encode_boxes, gaussian_radius
from seeanythingfar.utils.geometry import (bev_grid_centers, bev_to_metric, metric_to_bev, pixel_to_grid_sample,
                                           project_points_to_images, visible_in_image)

# regression outputs of each prediction head, in CODE order (see utils/box_ops.py)
REG_HEADS = (("center", 2), ("height", 1), ("dim", 3), ("rot", 2), ("vel", 2))


class PositionEmbedding(nn.Module):
    """Learned embedding of continuous BEV cell coordinates (normalised to [0, 1])."""

    def __init__(self, grid_hw: Tuple[int, int], dim: int):
        super().__init__()
        self.register_buffer("scale", torch.tensor([grid_hw[1], grid_hw[0]], dtype=torch.float32), persistent=False)
        self.mlp = nn.Sequential(nn.Linear(2, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim))

    def forward(self, cells: torch.Tensor) -> torch.Tensor:
        return self.mlp(cells / self.scale)


class FFN(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(hidden, dim))
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.drop(self.net(x)))


class LidarDecoderLayer(nn.Module):
    """Query self-attention, then cross-attention to the whole flattened BEV map (TransFusion-L)."""

    def __init__(self, dim: int, num_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.ffn = FFN(dim, ffn_dim, dropout)

    def forward(self, query, query_pos, key, key_pos):
        q = query + query_pos
        query = self.norm1(query + self.drop(self.self_attn(q, q, query, need_weights=False)[0]))
        attn = self.cross_attn(query + query_pos, key + key_pos, key, need_weights=False)[0]
        query = self.norm2(query + self.drop(attn))
        return self.ffn(query)


class ImageFusionLayer(nn.Module):
    """Cross Attention Fusion Module.

    Each 3D query attends only to its *own* keys: image features sampled in a local window
    around its projection in every camera (plus a learned null key, so a query visible in
    no camera still has something valid to attend to).
    """

    def __init__(self, dim: int, num_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.ffn = FFN(dim, ffn_dim, dropout)

    def forward(self, query, query_pos, keys, key_mask):
        """query/query_pos: (B, Q, D); keys: (B, Q, N_k, D); key_mask: (B, Q, N_k), True = ignore."""
        B, Q, D = query.shape
        q = query + query_pos
        query = self.norm1(query + self.drop(self.self_attn(q, q, query, need_weights=False)[0]))

        q = (query + query_pos).reshape(B * Q, 1, D)
        k = keys.reshape(B * Q, -1, D)
        attn = self.cross_attn(q, k, k, key_padding_mask=key_mask.reshape(B * Q, -1), need_weights=False)[0]
        query = self.norm2(query + self.drop(attn.reshape(B, Q, D)))
        return self.ffn(query)


class PredictionHead(nn.Module):
    """Detection Head (Prediction FFN): per-query box regression + classification."""

    def __init__(self, dim: int, num_classes: int, hidden: int = 64, init_bias: float = -2.19):
        super().__init__()
        outs = dict(REG_HEADS, cls=num_classes)
        self.heads = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, n))
            for name, n in outs.items()
        })
        nn.init.constant_(self.heads["cls"][-1].bias, init_bias)

    def forward(self, query: torch.Tensor, query_cells: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = {name: head(query) for name, head in self.heads.items()}
        out["center"] = out["center"] + query_cells  # offsets relative to the query's BEV cell
        out["code"] = torch.cat([out[name] for name, _ in REG_HEADS], dim=-1)  # (B, Q, CODE_SIZE)
        return out


class TransFusionHead(BaseFusionHead):
    """
    Two stage multi modal Fusion Head
    Stage 1: CenterPoint BEV heatmap (+ dense Z) -> top-K 3D object queries, refined by
             attention over the LiDAR BEV map
    Stage 2: queries projected through the calibration (geometric bridge) into every camera,
             cross-attending to local windows of the DINO feature map
    """
    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 128,
        num_classes: int = 10,
        num_queries: int = 200,                 # Top-K proposals to keep
        pc_range: Sequence[float] = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        bev_cell_size: Sequence[float] = (0.4, 0.4),  # metric size of one input BEV cell
        image_feature_scale: str = 'stride_8',
        image_channels: int = 256,
        num_heads: int = 8,
        ffn_channels: int = 256,
        dropout: float = 0.1,
        num_lidar_layers: int = 1,
        num_fusion_layers: int = 1,
        window_size: int = 3,                   # local k x k sampling window, in feature-map cells
        nms_kernel_size: int = 3,
        no_nms_classes: Sequence[int] = (8, 9),  # nuScenes pedestrian, traffic_cone: too small to NMS
        gaussian_overlap: float = 0.1,
        min_radius: int = 2,
        code_weights: Sequence[float] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2),
        loss_weights: Optional[Dict[str, float]] = None,
        assigner: Optional[Dict[str, float]] = None,
        score_threshold: float = 0.0,
        post_center_range: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.pc_range = list(pc_range)
        self.cell_size = list(bev_cell_size)
        self.grid_hw = (round((pc_range[4] - pc_range[1]) / bev_cell_size[1]),
                        round((pc_range[3] - pc_range[0]) / bev_cell_size[0]))
        self.target_scale = image_feature_scale
        self.nms_kernel_size = nms_kernel_size
        self.no_nms_classes = list(no_nms_classes)
        self.gaussian_overlap = gaussian_overlap
        self.min_radius = min_radius
        self.register_buffer("code_weights", torch.tensor(code_weights, dtype=torch.float32), persistent=False)
        assert len(code_weights) == CODE_SIZE
        self.loss_weights = dict(heatmap=1.0, z=0.25, cls=1.0, box=0.25)
        self.loss_weights.update(loss_weights or {})
        self.assigner = HungarianAssigner3D(**(assigner or {}))
        self.score_threshold = score_threshold
        self.post_center_range = list(post_center_range) if post_center_range is not None else None

        D = hidden_channels
        self.shared_conv = nn.Conv2d(in_channels, D, kernel_size=3, padding=1)

        # stage 1: CenterPoint proposal heads (BEV space)
        # object probability per (X, Y) cell and class
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(D, D, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(D), nn.ReLU(inplace=True),
            nn.Conv2d(D, num_classes, kernel_size=3, padding=1),
        )
        nn.init.constant_(self.heatmap_head[-1].bias, -2.19)
        # absolute z of the object centre, so queries are full 3D points before projection
        self.z_head = nn.Sequential(
            nn.Conv2d(D, D, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(D), nn.ReLU(inplace=True),
            nn.Conv2d(D, 1, kernel_size=3, padding=1),
        )
        self.class_encoding = nn.Linear(num_classes, D)
        self.pos_embed = PositionEmbedding(self.grid_hw, D)
        self.register_buffer("bev_pos", bev_grid_centers(*self.grid_hw), persistent=False)  # (HW, 2)

        self.lidar_layers = nn.ModuleList(
            [LidarDecoderLayer(D, num_heads, ffn_channels, dropout) for _ in range(num_lidar_layers)])
        self.fusion_layers = nn.ModuleList(
            [ImageFusionLayer(D, num_heads, ffn_channels, dropout) for _ in range(num_fusion_layers)])
        num_stages = max(1, num_lidar_layers + num_fusion_layers)
        self.pred_heads = nn.ModuleList([PredictionHead(D, num_classes) for _ in range(num_stages)])

        # stage 2: key construction for the image cross-attention
        self.img_proj = nn.Linear(image_channels, D)
        r = window_size // 2
        offsets = torch.tensor(list(itertools.product(range(-r, r + 1), repeat=2)), dtype=torch.float32)[:, [1, 0]]
        self.register_buffer("window_offsets", offsets, persistent=False)  # (K, 2) as (dx, dy) cells
        self.offset_embed = nn.Parameter(torch.zeros(offsets.shape[0], D))
        self.depth_embed = nn.Sequential(nn.Linear(1, D), nn.ReLU(inplace=True), nn.Linear(D, D))
        self.null_key = nn.Parameter(torch.zeros(1, D))
        nn.init.normal_(self.offset_embed, std=0.02)
        nn.init.normal_(self.null_key, std=0.02)

    # ------------------------------------------------------------------ forward
    def forward(self,
                bev_features: torch.Tensor,
                image_features: Optional[Dict[str, torch.Tensor]] = None,
                calibrations: Optional[Dict[str, torch.Tensor]] = None,
                ) -> Dict[str, torch.Tensor]:
        B, _, H, W = bev_features.shape
        assert (H, W) == self.grid_hw, f"BEV map {(H, W)} != head grid {self.grid_hw}; check pc_range / cell size"

        feat = self.shared_conv(bev_features)
        dense_heatmap = self.heatmap_head(feat)                    # [B, Classes, H, W] logits
        dense_z = self.z_head(feat)                                # [B, 1, H, W]

        # Extract the Top-K hottest cells to act as our 3D queries
        query_idx, query_labels, heat = self._get_topk_proposals(dense_heatmap)
        flat = feat.flatten(2).transpose(1, 2)                     # (B, HW, D)
        query = flat.gather(1, query_idx[..., None].expand(-1, -1, flat.shape[-1]))
        query = query + self.class_encoding(F.one_hot(query_labels, self.num_classes).float())
        query_cells = self.bev_pos[query_idx]                      # (B, K, 2)
        query_z = dense_z.flatten(1).detach().gather(1, query_idx)[..., None]  # (B, K, 1)
        query_heatmap_score = heat.gather(2, query_idx[:, None].expand(-1, self.num_classes, -1)).transpose(1, 2)

        stages: List[Dict[str, torch.Tensor]] = []
        head_i = 0
        bev_key_pos = self.pos_embed(self.bev_pos)[None]           # (1, HW, D)
        for layer in self.lidar_layers:
            query = layer(query, self.pos_embed(query_cells), flat, bev_key_pos)
            pred = self.pred_heads[head_i](query, query_cells)
            head_i += 1
            stages.append(pred)
            # refined 3D query positions for the next stage (no gradient through positions)
            query_cells, query_z = pred["center"].detach(), pred["height"].detach()

        use_images = image_features is not None and len(self.fusion_layers) > 0
        if use_images:
            img_feat = image_features[self.target_scale]           # (B, N_cam, C, h, w)
            for layer in self.fusion_layers:
                centers = torch.cat([bev_to_metric(query_cells, self.pc_range, self.cell_size), query_z], dim=-1)
                keys, key_mask = self._project_and_sample(centers, img_feat, calibrations)
                query = layer(query, self.pos_embed(query_cells), keys, key_mask)
                pred = self.pred_heads[head_i](query, query_cells)
                head_i += 1
                stages.append(pred)
                query_cells, query_z = pred["center"].detach(), pred["height"].detach()

        if not stages:  # no decoder layers at all: predict straight from the BEV query features
            stages.append(self.pred_heads[0](query, query_cells))

        return {
            "stages": stages,
            "dense_heatmap": dense_heatmap,
            "dense_z": dense_z,
            "query_labels": query_labels,
            "query_heatmap_score": query_heatmap_score,
        }

    def _get_topk_proposals(self, dense_heatmap: torch.Tensor):
        """Heatmap peaks (3x3 max-pool NMS) -> indices and classes of the top-K cells."""
        B, C, H, W = dense_heatmap.shape
        heat = dense_heatmap.detach().sigmoid()
        pad = self.nms_kernel_size // 2
        local_max = F.max_pool2d(heat, kernel_size=self.nms_kernel_size, stride=1, padding=pad)
        for c in self.no_nms_classes:
            if c < C:
                local_max[:, c] = heat[:, c]
        heat = (heat * (heat == local_max)).flatten(2)             # (B, C, HW)
        k = min(self.num_queries, C * H * W)
        top = heat.flatten(1).topk(k, dim=1).indices                # over all classes jointly
        return top % (H * W), top // (H * W), heat

    def _project_and_sample(self, centers: torch.Tensor, img_feat: torch.Tensor,
                            calibrations: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Geometric bridge: sample a k x k window of image features around each query's projection.

        Args:
            centers: (B, Q, 3) metric query centres in the LiDAR frame.
            img_feat: (B, N_cam, C, h, w).
        Returns:
            keys: (B, Q, N_cam * K + 1, D); key_mask: (B, Q, N_cam * K + 1), True = ignore.
        """
        B, N, C, h, w = img_feat.shape
        Q = centers.shape[1]
        image_hw = [int(v) for v in calibrations["image_hw"]]
        uv, depth = project_points_to_images(centers, calibrations["lidar2img"].to(centers.dtype))  # (B, N, Q, 2)
        valid = visible_in_image(uv, depth, image_hw)                                              # (B, N, Q)

        stride = uv.new_tensor([image_hw[1] / w, image_hw[0] / h])
        pts = uv.unsqueeze(3) + self.window_offsets * stride                                       # (B, N, Q, K, 2)
        K = pts.shape[3]
        grid = pixel_to_grid_sample(pts, image_hw).reshape(B * N, Q, K, 2)
        sampled = F.grid_sample(img_feat.reshape(B * N, C, h, w), grid, align_corners=False)      # (B*N, C, Q, K)
        sampled = sampled.reshape(B, N, C, Q, K).permute(0, 3, 1, 4, 2)                            # (B, Q, N, K, C)

        log_depth = depth.clamp(min=0.1).log().permute(0, 2, 1)[..., None]                         # (B, Q, N, 1)
        keys = self.img_proj(sampled) + self.offset_embed + self.depth_embed(log_depth)[:, :, :, None]
        keys = keys.reshape(B, Q, N * K, -1)
        mask = (~valid).permute(0, 2, 1)[..., None].expand(B, Q, N, K).reshape(B, Q, N * K)

        keys = torch.cat([keys, self.null_key.expand(B, Q, 1, -1)], dim=2)
        mask = torch.cat([mask, mask.new_zeros(B, Q, 1)], dim=2)
        return keys, mask

    # ------------------------------------------------------------------ training
    def loss(self, preds: Dict, gt_boxes: List[torch.Tensor], gt_labels: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            gt_boxes: list of (M_b, 9) metric boxes per sample; gt_labels: list of (M_b,) in [0, C).
        """
        dense = preds["dense_heatmap"]
        B, C, H, W = dense.shape
        heat_target, z_idx, z_target = self._dense_targets(gt_boxes, gt_labels, dense)
        loss_hm = gaussian_focal_loss(clip_sigmoid(dense), heat_target)
        if z_idx.numel():
            pred_z = preds["dense_z"].flatten(1)[z_idx[:, 0], z_idx[:, 1]]
            loss_z = F.l1_loss(pred_z, z_target)
        else:
            loss_z = preds["dense_z"].sum() * 0

        w = self.loss_weights
        total = w["heatmap"] * loss_hm + w["z"] * loss_z
        logs = {"loss_heatmap": loss_hm.detach(), "loss_z": loss_z.detach()}
        for s, stage in enumerate(preds["stages"]):
            loss_cls, loss_box, matched_iou = self._stage_loss(stage, gt_boxes, gt_labels)
            total = total + w["cls"] * loss_cls + w["box"] * loss_box
            tag = "" if s == len(preds["stages"]) - 1 else f"_aux{s}"
            logs[f"loss_cls{tag}"] = loss_cls.detach()
            logs[f"loss_box{tag}"] = loss_box.detach()
            logs[f"matched_iou{tag}"] = matched_iou
        logs["loss_det"] = total.detach()
        return total, logs

    def _dense_targets(self, gt_boxes, gt_labels, dense):
        B, C, H, W = dense.shape
        heat = dense.new_zeros(B, C, H, W)
        z_idx, z_val = [], []
        for b in range(B):
            boxes, labels = gt_boxes[b], gt_labels[b]
            if boxes.shape[0] == 0:
                continue
            cells = metric_to_bev(boxes[:, :2], self.pc_range, self.cell_size)
            for i in range(boxes.shape[0]):
                cx, cy = cells[i].tolist()
                if not (0 <= cx < W and 0 <= cy < H):
                    continue
                l_cells = float(boxes[i, 3]) / self.cell_size[0]
                w_cells = float(boxes[i, 4]) / self.cell_size[1]
                radius = max(self.min_radius, int(gaussian_radius(l_cells, w_cells, self.gaussian_overlap)))
                xi, yi = int(cx), int(cy)
                draw_gaussian(heat[b, int(labels[i])], (xi, yi), radius)
                z_idx.append((b, yi * W + xi))
                z_val.append(boxes[i, 2])
        z_idx = torch.tensor(z_idx, dtype=torch.long, device=dense.device).reshape(-1, 2)
        z_val = torch.stack(z_val) if z_val else dense.new_zeros(0)
        return heat, z_idx, z_val

    def _stage_loss(self, stage, gt_boxes, gt_labels):
        cls_logits, code = stage["cls"], stage["code"]
        B, Q, C = cls_logits.shape
        cls_target = torch.zeros_like(cls_logits)
        code_target = torch.zeros_like(code)
        code_weight = torch.zeros_like(code)
        num_pos, iou_sum = 0, code.new_zeros(())
        pred_boxes = decode_boxes(code.detach(), self.pc_range, self.cell_size)
        for b in range(B):
            if gt_boxes[b].shape[0] == 0:
                continue
            assigned, ious = self.assigner.assign(pred_boxes[b], cls_logits[b].detach(),
                                                  gt_boxes[b], gt_labels[b], self.pc_range)
            pos = torch.nonzero(assigned >= 0).squeeze(1)
            matched = assigned[pos]
            cls_target[b, pos, gt_labels[b][matched]] = 1.0
            code_target[b, pos] = encode_boxes(gt_boxes[b][matched], self.pc_range, self.cell_size)
            code_weight[b, pos] = 1.0
            num_pos += pos.numel()
            iou_sum = iou_sum + ious[pos].sum()
        norm = max(num_pos, 1)
        loss_cls = sigmoid_focal_loss(cls_logits, cls_target).sum() / norm
        loss_box = ((code - code_target).abs() * code_weight * self.code_weights).sum() / norm
        return loss_cls, loss_box, iou_sum / norm

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def decode(self, preds: Dict) -> List[Dict[str, torch.Tensor]]:
        """Final-stage predictions -> per-sample dicts of 'boxes' (M, 9), 'scores' (M,), 'labels' (M,)."""
        stage = preds["stages"][-1]
        one_hot = F.one_hot(preds["query_labels"], self.num_classes).float()
        score = stage["cls"].sigmoid() * preds["query_heatmap_score"] * one_hot
        scores, labels = score.max(-1)
        boxes = decode_boxes(stage["code"], self.pc_range, self.cell_size)

        keep = scores > self.score_threshold
        if self.post_center_range is not None:
            r = boxes.new_tensor(self.post_center_range)
            keep &= (boxes[..., :3] >= r[:3]).all(-1) & (boxes[..., :3] <= r[3:]).all(-1)
        out = []
        for b in range(boxes.shape[0]):
            k = keep[b]
            order = scores[b, k].argsort(descending=True)
            out.append({"boxes": boxes[b, k][order], "scores": scores[b, k][order], "labels": labels[b, k][order]})
        return out
