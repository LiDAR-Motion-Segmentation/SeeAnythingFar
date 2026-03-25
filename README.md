# SeeAnythingFar
Teaching Robots To See Far Away Obstacles!

## Slurm
- added local cuda setup for version 12.8 for compatibility
- code is in `slurm/neon/cuda_setup.sh`

## OpenPCDet
- currently I have used it as a backbone
- try to install openmmlab environment from their official website
```bash
# some additional dependency
conda activate openmmlab
pip install spconv-cu120

# for visualizations
pip install open3d
```
- setup the repo using
```
cd /src/seeanythingfar/openPCDet/
python3 setup.py develop
```

## Pretained weights
- currently used `pv_rcnn` weights for 3D object detection for far distance from [link](https://drive.google.com/file/d/1lIOq4Hxr0W3qsX83ilQv0nk1Cls6KAr-/view?usp=sharing)
- a copy of the orginal semantic-kitti dataset is there on the neon server
```python3
python3 demo.py \ 
--cfg_file cfgs/kitti_models/pv_rcnn.yaml \ 
--ckpt pv_rcnn_8369.pth \
--data_path /scratch/soumo_roy/semantic-kitty-dataset/dataset/sequences/00/velodyne/000000.bin 
```
![pic](./assets/pvrcnn.png)

## Slide Deck
- I am attaching the slide deck of the experiments done [link](https://docs.google.com/presentation/d/13FvgcBmLo90PO2pVOTtf5GvFLkXOykHQLE2QgfY7nAM/edit?usp=sharing)