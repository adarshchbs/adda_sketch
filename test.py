"""Test script to classify target data."""

import torch
import torch.nn as nn

from utils import make_variable
from preprocess import preprocess_image_1

def eval_tgt(encoder, classifier, data_loader, gpu_flag = False, gpu_name = 'cuda:0'):
    """Evaluation for target encoder by source classifier on target dataset."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0
    acc = 0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    correct = 0
    total = 0 

    # evaluate network
    for (images, labels) in data_loader.image_gen(split_type='val'):

        images = preprocess_image_1( array = images,
                                    split_type = 'val',
                                    use_gpu = gpu_flag, gpu_name= gpu_name  )

        labels = torch.tensor(labels,dtype=torch.long)

        if(gpu_flag == True):
            labels = labels.cuda(gpu_name)

        preds = classifier(encoder(images))
        loss += criterion(preds, labels).item()

        _, predicted = torch.max(preds.data,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        # pred_cls = preds.data.max(1)[1]
        # acc += pred_cls.eq(labels.data).cpu().sum()

    loss /= data_loader.size['val']
    # acc /= len(data_loader.dataset)
    acc = correct/total

    print("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))
