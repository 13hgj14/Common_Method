import argparse
import pdb
from datetime import datetime

import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.nn import functional as F
from tqdm import tqdm

from stn_分布式训练 import Net


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)
       
        self.stn_=Net()
        #加噪点
        self.sample=nn.Sequential(nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=5, stride=1, padding=2))
    def forward(self, x):
        #是否使用stn，空间变换网络，检测模型的性能
        x=self.stn_.stn(x)

        #是否添加噪点层
        # x=self.sample(x)+x

        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
def evaluate(model, gpu, test_loader, rank):
    model.eval()
    size = torch.tensor(0.).to(gpu)
    correct = torch.tensor(0.).to(gpu)
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(test_loader)):
            images = images.to(gpu)
            labels = labels.to(gpu)
            outputs = model(images)
            size += images.shape[0]
            correct += (outputs.argmax(1) == labels).type(torch.float).sum() 
    dist.reduce(size, 0, op=dist.ReduceOp.SUM) # 群体通信 reduce 操作 change to allreduce if Gloo
    dist.reduce(correct, 0, op=dist.ReduceOp.SUM) # 群体通信 reduce 操作 change to allreduce if Gloo
    if rank==0:
        print('Evaluate accuracy is {:.2f}'.format(correct / size))
def train(args):
    ########################################    N1    ####################################################################
    #初始化进程
    dist.init_process_group(backend='nccl', init_method='env://')    #
    #获取当前进程的序列号，所有机器中的进程
    args.rank = dist.get_rank()    #
    #设定当前进程运行的device
    torch.cuda.set_device(args.local_rank)
    ######################################################################################################################
    model = ConvNet()
    #送入gpu
    model.cuda()
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), 1e-2)
    # Wrap the model
    #######################################    N2    ########################
    #网络中的BN层对于分布式训练方法存在不兼容，会使得原本的BN层失效，因此需要进行替换
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # if args.rank==0:
    #     import pdb
    #     pdb.set_trace() 
    # print(1)                 #
    #使用DDP将网络包一下
    model = nn.parallel.DistributedDataParallel(model,find_unused_parameters=True) 
    
    from torch.cuda.amp import GradScaler

    #是否使用混合精度进行训练
    scaler = GradScaler(enabled=args.use_mix_precision)                   #
    #########################################################################
    # Data loading code
    train_dataset = torchvision.datasets.MNIST(root='../vision_datasets',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=False)
    ####################################    N3    #######################################
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)      #
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,                   #
                                               batch_size=args.batch_size,              #
                                               shuffle=False,                           #
                                               num_workers=0,                           #
                                               pin_memory=True,                         #
                                               sampler=train_sampler)                   #
    test_dataset = torchvision.datasets.MNIST(root='../vision_datasets',                        #
                                               train=False,                         #
                                               transform=transforms.ToTensor(),     #
                                               download=False)                       #
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)    #
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,                 #
                                               batch_size=args.batch_size,               #
                                               shuffle=False,                       #
                                               num_workers=0,                       #
                                               pin_memory=True,                     #
                                               sampler=test_sampler)
    #####################################################################################
    start = datetime.now()
    total_step = len(train_loader) # The number changes to orignal_length // args.world_size
    for epoch in range(args.epochs):
        ################    N4    ################
        #为当前的进程设置epoch，使得不同的进程下train_loader打乱的顺序不同，否则打乱顺序是相同
        train_loader.sampler.set_epoch(epoch)    #
        ##########################################
        model.train()
        for i, (images, labels) in enumerate(tqdm(train_loader)):
            images = images.cuda()
            labels = labels.cuda()
            # if args.rank==0:
            #     pdb.set_trace()
            # Forward pass
            ########################    N5    ################################
            #是否要混合精度下推理模型，以及计算损失
            with torch.cuda.amp.autocast(enabled=args.use_mix_precision):    #
                outputs = model(images)                                      #
                loss = criterion(outputs, labels)                            #
            ##################################################################  
            # Backward and optimize
            optimizer.zero_grad()
            ##############    N6    ##########
            scaler.scale(loss).backward()    #
            scaler.step(optimizer)           #
            scaler.update()                  #
            ##################################
            ################    N7    ####################
            if (i + 1) % 100 == 0 and args.rank == 0:    #
            ##############################################   
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.epochs, i + 1, total_step,
                                                                   loss.item()))
        #验证（这里采用的是分布式验证的方法）
        model.eval()
        #收集总共有多少个测试样本
        size = torch.tensor(0.).cuda()
        #准确率
        correct = torch.tensor(0.).cuda()
        with torch.no_grad():
            for i, (images, labels) in enumerate(tqdm(test_loader)):
                    images = images.cuda()
                    labels = labels.cuda()
                    outputs = model(images)
                    size += images.shape[0]
                    correct += (outputs.argmax(1) == labels).type(torch.float).sum() 
        #群体通信，主要用于想要将各个进程上的某个变量进行整合，op整合的方式
        dist.reduce(size, 0, op=dist.ReduceOp.SUM) # 群体通信 reduce 操作 change to allreduce if Gloo
        dist.reduce(correct, 0, op=dist.ReduceOp.SUM) # 群体通信 reduce 操作 change to allreduce if Gloo
        if args.rank==0:
                print('Evaluate accuracy is {:.2f}'.format(100*correct / size))            
    dist.destroy_process_group()    #                                       
    ############    N8    ###########
    if args.rank == 0:              #
    #################################
        print("Training complete in: " + str(datetime.now() - start))
def main():
    # print(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default=20, type=int, 
                        metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=64, type=int, 
                        metavar='N',
                        help='number of batchsize')   
    ##################################################################################
    #当前进程的序列号（当前机器中的第几个进程，因为训练的时候可能存在多个机器同时训练，机器由多个gpu组成）
    parser.add_argument("--local_rank", type=int,                                    #
                        help='rank in current node')  
    #是否使用混合精度训练
    parser.add_argument('--use_mix_precision', default=False,                        #
                        action='store_true', help="whether to use mix precision")    #
    ##################################################################################                  
    args = parser.parse_args()
    #################################
    train(args)
if __name__ == '__main__':
    import time
    main()