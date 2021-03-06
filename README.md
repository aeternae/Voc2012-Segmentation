# Voc2012-Segmentation
## Zhixiang Wang

Pascal voc 2012数据集可以用于分类、检测和分割。本项目利用U-Net模型完成语义分割任务。

### 1.U-Net概要

U-Net通俗来讲也是卷积神经网络的一种变形，主要其结构经论文作者画出来形似字母U（见图 1），因而得名U-Net。整个神经网络主要有两部分组成：收缩路径（contracting path）和扩展路径（expanding path）。搜索路径主要是用来捕捉图片中的上下文信息（context information），而与之相对称的扩展路径则是为了对图片中所需要分割出来的部分进行精准定位（localization）。U-Net诞生的一个主要前提是，很多时候深度学习的结构需要大量的sample和计算资源，但是U-Net基于FCN（Fully Convultional Neural Network：全卷积神经网络）进行改进，并且利用数据增强（data augmentation）可以对一些比较少样本的数据进行训练，特别是医学方面相关的数据（医学数据比一般我们所看到的图片及其他文本数据的获取成本更大，不论是时间还是资源的消耗），所以U-Net的出现对于深度学习用于较少样本的医学影像是很有帮助的。<br>
![](./pic/unet.jpeg)

### 2.U-Net网络结构及原理解析

就如前面所说的U-Net是基于FCN进行改进的，比较tricky的地方就是U-Net不是简单地像FCN那样子对图片进行encode和decode，U-Net为了能精准的定位，收缩路径上提取出来的高像素特征会在升采样（upsampling）过程中与新的特征图（feature map）进行结合，以最大程度的保留前面降采样（downsampling）过程一些重要的特征信息。而为了能使网络结构能更高效的运行，结构中是没有全连接层（fully connected layers），这样子可以很大程度上减少需要训练的参数，并得益于特殊的U形结构可以很好的保留图片中的所有信息。

收缩路径上是每两个 3 * 3 的卷积层（unpadded convolutional layers）后会跟一个2 * 2的最大池化层（Maxpooling layer: 步长为2），并且每个卷积层后面采用relu激活函数来对原始图片进行降采样操作，除此之外，每一次降采样都会增加一杯通道数（double the number of feature channel）。

在扩展路径的向上采样（deconvolution）中，每一步会有一个 2 * 2 的卷积层（激活函数也是relu）和一个两个 3 * 3 的卷积层，于此同时，每一步的升采样都会加入来自相对应收缩路径的特征图（经裁剪以保持相同的形状shape）。

在网络的最后一层是一个 1 * 1 的卷积层，通过这一操作可以将64通道的特征向量转换为所需要的分类结果的数量（例如2），最终，U-Net的整个网络一共有23层卷积层。U-Net有一个很重要的有点是其基本可以对任意形状大小的图片进行卷积操作，特别是任意大的图片。

### 3.项目展示
```
!python main.py
Using TensorFlow backend.

(?, 128, 128, 64)
(?, 128, 128, 64)
(?, 64, 64, 128)
(?, 64, 64, 128)
(?, 32, 32, 256)
(?, 32, 32, 256)
(?, 16, 16, 512)
(?, 16, 16, 512)
(?, 16, 16, 1024)
attention_block:
(?, 16, 16, 256)
(?, 16, 16, 256)
(?, 16, 16, 256)
(?, 16, 16, 1)
(?, 16, 16, 1)

(?, 32, 32, 1)
(?, 32, 32, 512)
-----------------
(?, 32, 32, 512)
(?, 32, 32, 512)
attention_block:
(?, 32, 32, 128)
(?, 32, 32, 128)
(?, 32, 32, 128)
(?, 32, 32, 1)
(?, 32, 32, 1)
(?, 64, 64, 1)
(?, 64, 64, 256)
-----------------
(?, 64, 64, 256)
(?, 64, 64, 256)
attention_block:
(?, 64, 64, 64)
(?, 64, 64, 64)
(?, 64, 64, 64)
(?, 64, 64, 1)
(?, 64, 64, 1)
(?, 128, 128, 1)
(?, 128, 128, 128)
-----------------
(?, 128, 128, 128)
(?, 128, 128, 128)
attention_block:
(?, 128, 128, 32)
(?, 128, 128, 32)
(?, 128, 128, 32)
(?, 128, 128, 1)
(?, 128, 128, 1)
(?, 256, 256, 1)
(?, 256, 256, 64)
-----------------
(?, 256, 256, 64)
(?, 256, 256, 64)
(?, 256, 256, 21)
(?, 256, 256, 21)

Epoch 1/100
2019-08-05 10:04:50.311084: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcudnn.so.7
172/183 [===========================>..] - ETA: 6s - loss: -0.0438 - MIoU: 0.0438next epoch
182/183 [============================>.] - ETA: 0s - loss: -0.0507 - MIoU: 0.0507next epoch
183/183 [==============================] - 180s 982ms/step - loss: -0.0514 - MIoU: 0.0514 - val_loss: -0.2479 - val_MIoU: 0.2479
Epoch 2/100
172/183 [===========================>..] - ETA: 6s - loss: -0.3298 - MIoU: 0.3298next epoch
182/183 [============================>.] - ETA: 0s - loss: -0.3330 - MIoU: 0.3330next epoch
183/183 [==============================] - 169s 923ms/step - loss: -0.3335 - MIoU: 0.3335 - val_loss: -0.4249 - val_MIoU: 0.4249
Epoch 3/100
172/183 [===========================>..] - ETA: 6s - loss: -0.4594 - MIoU: 0.4594next epoch
182/183 [============================>.] - ETA: 0s - loss: -0.4599 - MIoU: 0.4599next epoch
183/183 [==============================] - 170s 930ms/step - loss: -0.4600 - MIoU: 0.4600 - val_loss: -0.4793 - val_MIoU: 0.4793
Epoch 4/100
172/183 [===========================>..] - ETA: 6s - loss: -0.5158 - MIoU: 0.5158next epoch
182/183 [============================>.] - ETA: 0s - loss: -0.5153 - MIoU: 0.5153next epoch
183/183 [==============================] - 168s 915ms/step - loss: -0.5153 - MIoU: 0.5153 - val_loss: -0.5193 - val_MIoU: 0.5193
Epoch 5/100
172/183 [===========================>..] - ETA: 6s - loss: -0.5418 - MIoU: 0.5418next epoch
182/183 [============================>.] - ETA: 0s - loss: -0.5407 - MIoU: 0.5407next epoch
183/183 [==============================] - 166s 906ms/step - loss: -0.5407 - MIoU: 0.5407 - val_loss: -0.5367 - val_MIoU: 0.5367
...... ......
Epoch 00090: ReduceLROnPlateau reducing learning rate to 1.249999968422344e-06.
Epoch 91/100
172/183 [===========================>..] - ETA: 5s - loss: -0.6058 - MIoU: 0.6058next epoch
182/183 [============================>.] - ETA: 0s - loss: -0.6042 - MIoU: 0.6042next epoch
183/183 [==============================] - 159s 867ms/step - loss: -0.6041 - MIoU: 0.6041 - val_loss: -0.6024 - val_MIoU: 0.6024
Epoch 92/100
172/183 [===========================>..] - ETA: 5s - loss: -0.6058 - MIoU: 0.6058next epoch
182/183 [============================>.] - ETA: 0s - loss: -0.6042 - MIoU: 0.6042next epoch
183/183 [==============================] - 159s 866ms/step - loss: -0.6041 - MIoU: 0.6041 - val_loss: -0.6024 - val_MIoU: 0.6024
Epoch 93/100
172/183 [===========================>..] - ETA: 5s - loss: -0.6058 - MIoU: 0.6058next epoch
182/183 [============================>.] - ETA: 0s - loss: -0.6042 - MIoU: 0.6042next epoch
183/183 [==============================] - 159s 868ms/step - loss: -0.6041 - MIoU: 0.6041 - val_loss: -0.6024 - val_MIoU: 0.6024
Epoch 94/100
172/183 [===========================>..] - ETA: 5s - loss: -0.6058 - MIoU: 0.6058next epoch
182/183 [============================>.] - ETA: 0s - loss: -0.6042 - MIoU: 0.6042next epoch
183/183 [==============================] - 159s 871ms/step - loss: -0.6041 - MIoU: 0.6041 - val_loss: -0.6024 - val_MIoU: 0.6024
Epoch 95/100
172/183 [===========================>..] - ETA: 5s - loss: -0.6058 - MIoU: 0.6058next epoch
182/183 [============================>.] - ETA: 0s - loss: -0.6042 - MIoU: 0.6042next epoch
183/183 [==============================] - 160s 872ms/step - loss: -0.6041 - MIoU: 0.6041 - val_loss: -0.6024 - val_MIoU: 0.6024
Epoch 96/100
172/183 [===========================>..] - ETA: 5s - loss: -0.6058 - MIoU: 0.6058next epoch
182/183 [============================>.] - ETA: 0s - loss: -0.6042 - MIoU: 0.6042next epoch
183/183 [==============================] - 159s 867ms/step - loss: -0.6041 - MIoU: 0.6041 - val_loss: -0.6024 - val_MIoU: 0.6024
Epoch 97/100
172/183 [===========================>..] - ETA: 5s - loss: -0.6058 - MIoU: 0.6058next epoch
182/183 [============================>.] - ETA: 0s - loss: -0.6042 - MIoU: 0.6042next epoch
183/183 [==============================] - 160s 875ms/step - loss: -0.6041 - MIoU: 0.6041 - val_loss: -0.6024 - val_MIoU: 0.6024
Epoch 98/100
172/183 [===========================>..] - ETA: 5s - loss: -0.6058 - MIoU: 0.6058next epoch
182/183 [============================>.] - ETA: 0s - loss: -0.6042 - MIoU: 0.6042next epoch
183/183 [==============================] - 159s 870ms/step - loss: -0.6041 - MIoU: 0.6041 - val_loss: -0.6024 - val_MIoU: 0.6024
Epoch 99/100
172/183 [===========================>..] - ETA: 5s - loss: -0.6058 - MIoU: 0.6058next epoch
182/183 [============================>.] - ETA: 0s - loss: -0.6042 - MIoU: 0.6042next epoch
183/183 [==============================] - 159s 871ms/step - loss: -0.6041 - MIoU: 0.6041 - val_loss: -0.6024 - val_MIoU: 0.6024
Epoch 100/100
172/183 [===========================>..] - ETA: 5s - loss: -0.6058 - MIoU: 0.6058next epoch
182/183 [============================>.] - ETA: 0s - loss: -0.6042 - MIoU: 0.6042next epoch
183/183 [==============================] - 159s 871ms/step - loss: -0.6041 - MIoU: 0.6041 - val_loss: -0.6024 - val_MIoU: 0.6024
