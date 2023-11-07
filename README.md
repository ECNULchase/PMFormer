# Point Mask Transformer for Outdoor Point Cloud Semantic Segmentation (CVM)

## Introduction
This is an implementation of the Point Mask Transformer model for outdoor point cloud semantic segmentation. 

## requirements
- [spconv](https://github.com/traveller59/spconv)
- [torchscatter](https://github.com/rusty1s/pytorch_scatter)



## Data Preparation

### SemanticKITTI
Please download the files from the [SemanticKITTI website](http://semantic-kitti.org/dataset.html).

Organize the data in the following directory structure:
```
./dataset/
├── 
├── ...
└── SemanticKitti/
    ├──sequences
        ├── 00/           
        │   ├── velodyne/	
        |   |	├── 000000.bin
        |   |	├── 000001.bin
        |   |	└── ...
        │   └── labels/ 
        |   |   ├── 000000.label
        |   |   ├── 000001.label
        |   |   └── ...
        |   └── image_2/ 
        |   |   ├── 000000.png
        |   |   ├── 000001.png
        |   |   └── ...
        |   calib.txt
        ├── 08/ # for validation
        ├── 11/ # 11-21 for testing
        └── 21/
	    └── ...
```

## Training
### SemanticKITTI
Training on a single GPU:
```bash
python tools/train.py configs/benchmarks/mmsegmentation/semantic_kitti/pmformer.py --work_dir work_dirs/logs
```
Training on multiple GPUs:
```bash
./tools/dist_train.sh configs/benchmarks/mmsegmentation/semantic_kitti/pmformer.py 8 --work_dir work_dirs/logs
```
The output will be written to work_dirs/logs.
## Testing
Run testing:
```bash
python tools/test.py configs/benchmarks/mmsegmentation/semantic_kitti/pmformer.py checkpoint_path
```

## acknowledgement
- [mmselfsup](https://github.com/open-mmlab/mmselfsup)
