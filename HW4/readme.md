## Directory

```
|--HW4
|  |--checkpoints                     // pre-trained checkpoints
|  |--data                            // CIFAR10 data directory
|  |--images                          // generated sample images
|  |--net                             // network codes for DCGAN, WGAN, and ACGAN
|  |--train.py                        // training code
|  |--test.py                         // testing code
```

How to run:
1. For training, please run the following code, and network argument with dcgan, wgan, or acgan.
```
python train.py --network dcgan --epochs 100 --batch-size 64 --lr 1e-4
```
3. For testing, please run the following code.
```
python test.py --network dcgan
```