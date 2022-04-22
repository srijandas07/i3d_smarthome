import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
# os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='rgb', type=str, help='rgb or flow')
parser.add_argument('--path', default='', type=str)
parser.add_argument('--protocol', default='CS', type=str)
parser.add_argument('--root', default='', type=str)

args = parser.parse_args()


import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import videotransforms
from pytorch_i3d import InceptionI3d
from dataset_test import *


def run(init_lr=0.1, max_steps=1, mode='rgb', root='', batch_size=3, path='', protocol='CS'):
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    if protocol == 'CS':
       num_classes = 31
    else:
       num_classes = 19

    dataset = Dataset('./labels/test_Labels_'+protocol+'.csv', 'test', root, 'rgb', test_transforms, protocol)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=36, pin_memory=True)

    dataloaders = {'test': dataloader}
    datasets = {'test': dataset}
    
    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load('models/flow_imagenet.pt'))
    else:
        i3d = InceptionI3d(400, in_channels=3)

    i3d.replace_logits(num_classes)
    i3d.load_state_dict(torch.load('./models/{}'.format(path)))
    i3d.cuda()
    i3d = nn.DataParallel(i3d)

    lr = init_lr
    optimizer = optim.SGD(i3d.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
    pred_arr = np.zeros((len(dataset), num_classes))

    steps = 0
    # train it
    while steps < max_steps:
        print ('Step {}/{}'.format(steps, max_steps))
        print ('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['test']:
            if phase == 'train':
                i3d.train(True)
            else:
                i3d.train(False)  # Set model to evaluate mode
            tot_acc = 0.0
            num_iter = 0
            optimizer.zero_grad()
            bal_dict = Bal_Dict()
            acount = 0
            # Iterate over data.
            for data in dataloaders[phase]:
                num_iter += 1
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                inputs = Variable(inputs.cuda())
                t = inputs.size(2)
                labels = Variable(labels.cuda())

                per_frame_logits = i3d(inputs)
                y_true = torch.max(per_frame_logits, dim=1)[1]

                for count in range(len(y_true.squeeze().tolist())):
                    l = per_frame_logits[count,:].cpu().detach().numpy()
                    pred_arr[acount,:] = l
                    acount+=1

                acc = calculate_accuracy(per_frame_logits, torch.max(labels, dim=1)[1])
                bal_dict.bal_update(per_frame_logits, torch.max(labels, dim=1)[1])
                tot_acc += acc
            steps += 1
            np.save("pred_arr.txt", pred_arr)
            if phase == 'test':
                print ('{} Acc: {:.4f}, Bal_acc: {:.4f}'.format(phase, tot_acc/num_iter, bal_dict.bal_score()))


if __name__ == '__main__':
    # need to add argparse
    run(mode=args.mode, root=args.root, path=args.path, batch_size=8, protocol=args.protocol)
