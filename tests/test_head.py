import torch

from seeanythingfar.datasets.collate import collate_fn
from seeanythingfar.datasets.synthetic import SyntheticDataset
from seeanythingfar.models.backbone_3d.sonata import SonataEncoder
from seeanythingfar.models.fusion_heads.transfusion import TransFusionHead

PC = [-25.6, -25.6, -5.0, 25.6, 25.6, 3.0]


def _setup(with_images=True):
    torch.manual_seed(0)
    ds = SyntheticDataset(length=2, pc_range=PC, image_size=(64, 96) if with_images else None)
    batch = collate_fn([ds[0], ds[1]])
    enc = SonataEncoder(backend="mlp", embed_dim=32, pc_range=PC, bev_voxel_size=[0.4, 0.4], bev_stride=2)
    head = TransFusionHead(in_channels=32, hidden_channels=32, num_queries=30, pc_range=PC,
                           bev_cell_size=enc.bev_cell_size, image_channels=16, num_heads=4, ffn_channels=64)
    return batch, enc, head


def _image_feats(batch):
    B, N = batch["images"].shape[:2]
    h, w = batch["image_hw"]
    return {"stride_8": torch.randn(B, N, 16, h // 8, w // 8, requires_grad=True)}


def test_forward_shapes_and_backward():
    batch, enc, head = _setup()
    tokens, bev = enc(batch["points"])
    assert tokens.shape == (sum(p.shape[0] for p in batch["points"]), 32)
    assert bev.shape == (2, 32, 64, 64)
    feats = _image_feats(batch)
    calib = {"lidar2img": batch["lidar2img"], "image_hw": batch["image_hw"]}
    preds = head(bev, feats, calib)
    assert len(preds["stages"]) == 2                     # lidar stage + fusion stage
    assert preds["stages"][-1]["code"].shape == (2, 30, 10)
    loss, logs = head.loss(preds, batch["gt_boxes"], batch["gt_labels"])
    assert torch.isfinite(loss)
    loss.backward()
    assert feats["stride_8"].grad is not None and feats["stride_8"].grad.abs().sum() > 0
    assert head.null_key.grad is not None


def test_lidar_only_and_decode():
    batch, enc, head = _setup(with_images=False)
    _, bev = enc(batch["points"])
    preds = head(bev)
    assert len(preds["stages"]) == 1
    out = head.decode(preds)
    assert len(out) == 2 and out[0]["boxes"].shape[1] == 9
    assert (out[0]["scores"][:-1] >= out[0]["scores"][1:]).all()


def test_queries_invisible_everywhere_do_not_nan():
    batch, enc, head = _setup()
    _, bev = enc(batch["points"])
    calib = {"lidar2img": torch.zeros_like(batch["lidar2img"]), "image_hw": batch["image_hw"]}  # nothing projects
    preds = head(bev, _image_feats(batch), calib)
    assert torch.isfinite(preds["stages"][-1]["code"]).all()


def test_overfits_one_batch():
    batch, enc, head = _setup(with_images=False)
    params = list(enc.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=2e-3)
    losses = []
    for _ in range(40):
        _, bev = enc(batch["points"])
        loss, _ = head.loss(head(bev), batch["gt_boxes"], batch["gt_labels"])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0]
