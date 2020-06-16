# coding=utf-8
from fileio import generateFaceRS, readRiIndex
from torch.autograd import Variable
from torchvision import models
import torch.nn as nn
import numpy as np
import torch


def extra_feature(img_array, net, use_gpu):
    print('Start DenseNet feature preprocessing...')

    for i in range(img_array.shape[0]):

        eachImage = img_array[i]

        # 二维变成三维
        singleChannelImage = np.reshape(eachImage, (1, 128, 128))

        # 单通道变成三通道
        multiChannelImage = np.concatenate(
            (singleChannelImage, singleChannelImage, singleChannelImage), axis=0
        )

        # numpy array转torch tensor
        multiChannelImage_tensor = torch.from_numpy(multiChannelImage)

        x = Variable(torch.unsqueeze(
            multiChannelImage_tensor, dim=0).float(), requires_grad=False
        )

        if use_gpu:
            x = x.cuda()
            net = net.cuda()

        y = net(x).cpu()
        y = y.data.numpy()

        if (i == 0):
            face = y
        elif (i >= 1):
            face = np.concatenate((face, y), axis=0)

    print('Shape of the result:' + str(face.shape))

    return face


def run(method_generateFaceRS='densenet'):
    ri, index = readRiIndex()

    densenet201 = models.densenet201(pretrained=True)

    densenet201.fc = nn.Linear(2048, 2048)
    torch.nn.init.eye(densenet201.fc.weight)

    for param in densenet201.parameters():
        param.requires_grad = False

    use_gpu = torch.cuda.is_available()

    X = extra_feature(ri, densenet201, use_gpu)

    X = np.concatenate([index, X], axis=1)

    # 进行DenseNet特征处理后，划分数据集，并将结果写到目录里
    generateFaceRS(X, method_generateFaceRS)

