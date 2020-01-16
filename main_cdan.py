"""Main script for ADDA."""
import os
import numpy as np
import torch

import params
from pretrain import eval_src, train_src
from resnet_18 import ResNetClassifier, ResNetEncoder
from discriminator import RandomLayer, AdversarialNetwork
from utils import init_model, init_random_seed
from test import eval_tgt
from adopt_cdan import train_target
from image_loader import image_loader

from torchvision import models


resnet = models.resnet50(pretrained=True)
resnet1 = models.resnet50(pretrained=True)


encoder = torch.nn.Sequential(*(list(resnet.children())[:-1]))
encoder1 = torch.nn.Sequential(*(list(resnet1.children())[:-1]))

if __name__ == '__main__':
    # init random seed
    gpu_name = 'cuda:1'
    init_random_seed(params.manual_seed)

    path_sketchy = '/home/adarsh/project/adda_sketch/dataset/sketches/'
    path_quickdraw = '/home/adarsh/project/adda_sketch/dataset/QuickDraw_sketches_final/'
    path_class_list = '/home/adarsh/project/adda_sketch/common_class_list.txt'
    
    

    class_list = np.loadtxt(path_class_list,dtype='str')
    # load dataset
    source_loader = image_loader(parent_folder_path = path_sketchy,
                         folder_list= class_list,
                         split= [0.8,0.2,0] )
    # print(source_loader.size_total)
    src_data_loader = source_loader.image_gen(split_type='train')
    src_data_loader_eval = source_loader.image_gen(split_type='val')

    target_loader = image_loader(parent_folder_path = path_quickdraw,
                                folder_list = class_list,
                                split = [0.03, 0.01, 0] )
    

    
    # load models

    src_encoder = ResNetEncoder(encoder)
    src_encoder.cuda(gpu_name)
    src_classifier = ResNetClassifier()
    src_classifier.cuda(gpu_name)
    tgt_encoder = ResNetEncoder(encoder1)
    tgt_encoder.cuda(gpu_name)

    random_layer = RandomLayer([2048,87])
    random_layer.cuda(gpu_name)
    critic = AdversarialNetwork(1024,1024)
    critic.cuda(gpu_name)

    


    if(os.path.exists(params.src_encoder_restore) and 
        os.path.exists(params.src_classifier_restore)):
        src_encoder.load_state_dict(torch.load(params.src_encoder_restore))
        src_classifier.load_state_dict(torch.load(params.src_classifier_restore))

    else:
        src_encoder, src_classifier = train_src( src_encoder,
                                                src_classifier,
                                                source_loader, gpu_flag = True)



    

    # eval source model
    # print("=== Evaluating classifier for source domain ===")
    # eval_src(src_encoder, src_classifier, source_loader, gpu_flag = True)
    # print("=== Evaluating target encoder for source domain ===")
    # eval_src(tgt_encoder, src_classifier, source_loader, gpu_flag = True)

   
       
    # train target encoder by GAN
    tgt_encoder.load_state_dict(src_encoder.state_dict())


    if(os.path.exists(params.tgt_encoder_restore)):
        tgt_encoder.load_state_dict(torch.load(params.tgt_encoder_restore))

    else:
        tgt_encoder = train_target( src_encoder, src_classifier, critic, random_layer,
                            source_loader, target_loader, gpu_flag = True, gpu_name= gpu_name)


    
    # eval target encoder on test set of target dataset
    print("=== Evaluating classifier for encoded target domain ===")
    # print(">>> source only <<<")
    # eval_tgt(src_encoder, src_classifier, target_loader, gpu_flag = True, gpu_name= gpu_name)
    # print(">>> domain adaption <<<")
    # eval_tgt(tgt_encoder, src_classifier, target_loader, gpu_flag = True, gpu_name= gpu_name)


