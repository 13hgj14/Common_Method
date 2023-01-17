# License: BSD
# Author: Ghassen Hamrouni

from __future__ import print_function

import pdb

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as distributed
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed as data_distributed
import torchvision
from torch.cuda.amp import GradScaler
from torch.nn import SyncBatchNorm, parallel
from torchvision import datasets, transforms
from tqdm import tqdm

plt.ion()   # interactive mode
import os
import random

import numpy as np
from six.moves import urllib
from torch.nn import DataParallel

from swinTransformer import swin_tiny_model


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        #加噪点
        self.sample=nn.Sequential(nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=5, stride=1, padding=2),
            nn.ReLU6())
    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(),align_corners=True)
        x = F.grid_sample(x, grid,align_corners=True)

     
        #是否添加噪点层,会略微提升模型性能，
        x=self.sample(x)+x
        return x

    def forward(self, x):
        
        # transform the input
        #对输入图像放射变换
        x = self.stn(x)
        #随后再利用变换后的图像训练分类模型
        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)

def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

# We want to visualize the output of the spatial transformers layer
# after the training, we visualize a batch of input images and
# the corresponding transformed batch using STN.

def setup_seed(args):
    from torch.backends import cudnn
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cudnn.benchmark = args.benchmark

def visualize_stn(test_loader,model):
    with torch.no_grad():
        # Get a batch of training data
       
        data = next(iter(test_loader))[0]
        input_tensor = data.cpu()
        net=Net()
        net.load_state_dict(model.module.state_dict())
        # pdb.set_trace()
        transformed_input_tensor = net.stn(data).cpu()
        in_grid = convert_image_np(
            torchvision.utils.make_grid(input_tensor))

        out_grid = convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor))

        # Plot the results side-by-side
        # pdb.set_trace()
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Dataset Images')

        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')




def main(args):
    #初始化进程
    distributed.init_process_group(backend='nccl',init_method='env://')
    #获取当前总进程序列号
    args.rank=distributed.get_rank()
    #设定随机种子
    setup_seed(args)
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)
    #设定当前进程gpu环境
    torch.cuda.set_device(args.local_rank)
    #创建模型
    net=Net()
    # net=swin_tiny_model(path="/mnt/lustre/GPU3/home/hugaojie/TORCH_LEARNING/save_model_weight/swin_tiny_patch4_window7_224.pth")
    # num_features=net.head.in_features
    # net.head=nn.Linear(num_features, 10)
    #将模型送入到gpu中
    net.cuda()
    #替换模型中的bn层
    model=SyncBatchNorm.convert_sync_batchnorm(net)
    #创建DDP模型
    model=parallel.DistributedDataParallel(model,find_unused_parameters=True)
    #创建数据集
    # Training dataset
    train_dataset=datasets.MNIST(root='../vision_datasets', train=True, download=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
    #创建分布式训练样本集
    train_sampler=data_distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,sampler=train_sampler)
    # Test dataset
    test_datasets=datasets.MNIST(root='../vision_datasets', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))
    test_sampler=data_distributed.DistributedSampler(test_datasets)
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=args.batch_size, sampler=test_sampler)
    #创建优化器SGD的效果最好，精度能够达到更高的点
    # optimizer = optim.SGD(model.parameters(), lr=0.01)
    # optimizer = optim.SGD(model.parameters(), lr=0.01,momentum=0.9)
    optimizer = optim.SGD(model.parameters(), lr=0.01,momentum=0.9,nesterov=True)
    # optimizer = optim.Adam(model.parameters(), lr=0.01)
    # optimizer = optim.AdamW(model.parameters(), lr=0.01)
    #创建混合精度函数
    from torch.cuda.amp import GradScaler
    scaler=GradScaler(enabled=args.use_mix)
    #训练
    from datetime import datetime
    start=datetime.now()
    best_acc=torch.tensor(0.)
    for epoch in range(args.epochs):
        model.train()
        #为每个进程设定当前的epoch
        train_sampler.set_epoch(epoch)
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            # if args.rank==0:
            #     pdb.set_trace()
            # data=data.repeat(1,3,1,1)
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            #在混合精度函数下推算模型与计算损失
            with torch.cuda.amp.autocast(enabled=args.use_mix):
                output = model(data)
                loss = F.nll_loss(output, target,reduction='mean')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            #在进程0中打印
            if args.rank==0:
                if batch_idx % 200 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))
        #分布式测试
        with torch.no_grad():
            model.eval()
            test_loss = torch.tensor(0.).cuda()
            correct = torch.tensor(0.).cuda()
            size=torch.tensor(0.).cuda()
            for i,(data, target) in enumerate(tqdm(test_loader)):
                # data=data.repeat(1,3,1,1)
                data, target = data.cuda(), target.cuda()
                output = model(data)
                # sum up batch loss
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                # get the index of the max log-probability
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()
                size+=data.shape[0]
            #多进程变量求和
           
            distributed.reduce(test_loss,0,op=distributed.ReduceOp.SUM)
            distributed.reduce(correct,0,op=distributed.ReduceOp.SUM)
            distributed.reduce(size,0,op=distributed.ReduceOp.SUM)
            if args.rank==0:
                print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%)\n'
                    .format(test_loss/size, 100. * correct/size))
                if (100. * correct/size)>best_acc:
                    best_acc=100. * correct/size
                    best_epoch=epoch
                    best_model=model
    #消除进程
    distributed.destroy_process_group()
    if args.rank == 0:              
        #################################
        print("Training complete in: " + str(datetime.now() - start))
        print('best model epoch:{} best accuaacy:{:.2f}%'.format(best_epoch+1,best_acc))
        print('测试图片总数量:'+str(size.item()))
        visualize_stn(test_loader,best_model)
        plt.ioff()
        plt.show()
        plt.savefig('./stn.png')
        

if __name__=='__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
    import argparse
    parse=argparse.ArgumentParser()
    parse.add_argument('--local_rank',type=int,help='获取当前进程的序列号')
    parse.add_argument('--use_mix',default=False,type=bool,help='是否使用混合精度训练模型')
    parse.add_argument('--epochs',default=20,type=int,help='训练的总轮数')
    parse.add_argument('--batch_size',default=32,type=int,help='批次大小')
    parse.add_argument('--seed',default=99,type=int,help='设定随机种子，确保每次初始化结果都相同')
    parse.add_argument('--benchmark',default=False,type=bool,help='是否对conv进行优化，提升效率，降低精度')
    args=parse.parse_args()
    main(args)


