import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
#os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='rgb', type=str, help='rgb or flow')
parser.add_argument('--save_model', default='weights/', type=str)
parser.add_argument('--root', default='', type=str)
parser.add_argument('--protocol', default='CS', type=str)

args = parser.parse_args()


import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from torchvision import datasets, transforms
import videotransforms

from pytorch_i3d import InceptionI3d
from dataset import *


def run(init_lr=0.01, max_steps=100, mode='rgb', root='', batch_size=16, save_model='weights/', protocol='CS'):
    # setup dataset
    train_transforms = transforms.Compose([videotransforms.CenterCrop(224)])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])
    original_protocol = protocol
    if protocol == 'CS':
       protocol = 'sub_'+protocol
       num_classes = 31
    else:
       num_classes = 19

    dataset = Dataset('/data/stars/user/sdas/smarthomes_data/splits/train_'+protocol+'.txt', 'train', root, 'rgb', train_transforms, original_protocol)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=36, pin_memory=True)

    val_dataset = Dataset('/data/stars/user/sdas/smarthomes_data/splits/validation_'+protocol+'.txt', 'val', root, 'rgb', test_transforms, original_protocol)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=36, pin_memory=True)    

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}
    
    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load('models/flow_imagenet.pt'))
    else:
        i3d = InceptionI3d(400, in_channels=3)
        i3d.load_state_dict(torch.load('models/rgb_imagenet.pt'))
    i3d.replace_logits(num_classes)
    
    i3d.cuda()
    i3d = nn.DataParallel(i3d)

    lr = init_lr
    optimizer = optim.SGD(i3d.parameters(), lr=lr, momentum=0.9)
    lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, verbose=True)

    num_steps_per_update = 1 # accum gradient
    steps = 0
    # train it
    while steps < max_steps:
        print ('Step {}/{}'.format(steps, max_steps))
        print ('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                i3d.train(True)
            else:
                i3d.train(False)  # Set model to evaluate mode

            tot_loss = 0.0
            tot_loc_loss = 0.0
            tot_cls_loss = 0.0
            tot_acc = 0.0
            num_iter = 0
            optimizer.zero_grad()
            
            # Iterate over data.
            for data in dataloaders[phase]:
                num_iter += 1
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())

                per_frame_logits = i3d(inputs)
                criterion=nn.CrossEntropyLoss().cuda()
                cls_loss = criterion(per_frame_logits, torch.max(labels, dim=1)[1].long())
                tot_cls_loss += cls_loss.data

                loss = cls_loss
                tot_loss += loss.data
                loss.backward()
                acc = calculate_accuracy(per_frame_logits, torch.max(labels, dim=1)[1])
                tot_acc += acc
                if phase == 'train':
                    optimizer.step()
                    optimizer.zero_grad()#lr_sched.step()

            if phase == 'train':
                print ('{} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}, Acc: {:.4f}'.format(phase, tot_loc_loss/num_iter, tot_cls_loss/num_iter, tot_loss/num_iter, tot_acc/num_iter))
                # save model
                torch.save(i3d.module.state_dict(), save_model+str(steps).zfill(6)+'.pt')
                tot_loss = tot_loc_loss = tot_cls_loss = tot_acc = 0.
                steps += 1
            if phase == 'val':
                lr_sched.step(tot_cls_loss/num_iter)
                print ('{} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}, Acc: {:.4f}'.format(phase, tot_loc_loss/num_iter, tot_cls_loss/num_iter, (tot_loss*num_steps_per_update)/num_iter, tot_acc/num_iter))
    

if __name__ == '__main__':
    # need to add argparse
    run(mode=args.mode, root=args.root, batch_size=16, save_model=args.save_model, protocol=args.protocol)
