# pytorch-deeplab-xception

Train your own dataset with deeplab v3+. 



### Introduction
This is a PyTorch(0.4.1) implementation of [DeepLab-V3-Plus](https://arxiv.org/pdf/1802.02611). It
can use Modified Aligned Xception and ResNet as backbone. Currently, we train DeepLab V3 Plus
using Pascal VOC 2012, SBD and Cityscapes datasets.

### Installation
The code was tested with Anaconda and Python 3.6. After installing the Anaconda environment:

0. Clone the repo:
    ```Shell
    git clone https://github.com/jfzhang95/pytorch-deeplab-xception.git
    cd pytorch-deeplab-xception
    ```

1. Install dependencies:

    For PyTorch dependency, see [pytorch.org](https://pytorch.org/) for more details. Recommand Ubuntu 18.04 with Cuda 10.1.

    For custom dependencies:
    ```Shell
    pip install matplotlib pillow tensorboardX tqdm
    ```
### Training

Follow steps below to train your model:

0. Create a datasets/tooth folder/.
```
ImageSets
|_Segmentation
   |_test.txt
     train.txt
     trainval.txt
     val.txt
JPEGImages
|_****.jpg
SegmentationClass
|_****.png
```

1. Configure your dataset path in [mypath.py](https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/mypath.py). 
```
12 elif dataset=='tooth':
      return 'datasets/tooth/'
```
2. Create your own dataset class in dataloaders/datasets/tooth.py,that is, Copy pascal.py and change NUM_CLASSES and dataset dir.
```
14 NUM_CLASSES = 2 # including background
18 base_dir=Path.db_root_dir('tooth')
```
3. Config your own dataloaders in dataloaders/__init__.py.
```
20     elif args.dataset == 'tooth':
        train_set = tooth.VOCSegmentation(args, split='train')
        val_set = tooth.VOCSegmentation(args, split='val')

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None

        return train_loader, val_loader, test_loader, num_class
```
4. Create label visualization in dataloaders/utils.py.
```
27     elif dataset=='tooth':
        n_classes=2
        label_colours=get_tooth_labels()
       
       def get_tooth_labels():
             return np.asarray([[0,0,0],[255,255,255]])
       # as the standard VOC dataset format, label is {0,1} in your own dataset rather {0,255}, this function can help to show {0,255} binary masks. 
```
5. Config train.py
```
275 'tooth':500
291 'tooth':0.01
```

6. To train deeplabv3+ using Pascal VOC dataset and ResNet as backbone: (only single GPU, tested)
    ```
   python train.py --backbone mobilenet  --workers 1 --batch-size 8 --gpu-ids 0 --checkname deeplab-mobilenet

### Testing
```
python test.py --in-path test_img_folder --ckpt path/to/model_best.pth.tar --backbone mobilenet
```

### Acknowledgement
[PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding)

[Synchronized-BatchNorm-PyTorch](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch)

[drn](https://github.com/fyu/drn)
