"""Main script for ADDA."""
import os
import numpy as np
import torch

import params
from pretrain import eval_src, train_src
from resnet_18 import ResNetClassifier, ResNetEncoder
from discriminator import Discriminator
from utils import init_model, init_random_seed
from test import eval_tgt
from adopt import train_target
from image_loader import image_loader

from torchvision import models


resnet = models.resnet18(pretrained=True)
resnet1 = models.resnet18(pretrained=True)


encoder = torch.nn.Sequential(*(list(resnet.children())[:-1]))
encoder1 = torch.nn.Sequential(*(list(resnet1.children())[:-1]))

if __name__ == '__main__':
    # init random seed
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
                                split = [0.05, 0.1, 0] )
    

    # tgt_data_loader = get_data_loader(params.tgt_dataset)
    # tgt_data_loader_eval = get_data_loader(params.tgt_dataset, train=False)

    # load models

    src_encoder = ResNetEncoder(encoder)
    src_encoder.cuda()
    src_classifier = ResNetClassifier()
    src_classifier.cuda()
    tgt_encoder = ResNetEncoder(encoder1)
    tgt_encoder.cuda()
    critic = Discriminator(input_dim=params.d_input_dims,
                                      hidden_dim=params.d_hidden_dims,
                                      output_dim=params.d_output_dims)
    critic.cuda()

    # src_encoder = init_model(net=ResNetEncoder(), restore = params.src_encoder_restore)
    # src_classifier = init_model(net=ResNetClassifier(),
    #                             restore=params.src_classifier_restore)
    # tgt_encoder = init_model(net=ResNetEncoder(),
    #                          restore=params.tgt_encoder_restore)
    # critic = init_model(Discriminator(input_dim=params.d_input_dims,
    #                                   hidden_dim=params.d_hidden_dims,
    #                                   output_dim=params.d_output_dims),
    #                     restore=params.d_model_restore)

    # train source model
    # print("=== Training classifier for source domain ===")
    # print(">>> Source Encoder <<<")
    # print(src_encoder)
    # print(">>> Source Classifier <<<")
    # print(src_classifier)
    # print("=== Evaluating classifier for source domain with resnet weights ===")
    # eval_src(src_encoder, src_classifier, source_loader, gpu_flag = True)

    if(os.path.exists(params.src_encoder_restore) and 
        os.path.exists(params.src_classifier_restore)):
        src_encoder.load_state_dict(torch.load(params.src_encoder_restore))
        src_classifier.load_state_dict(torch.load(params.src_classifier_restore))

    else:
        src_encoder, src_classifier = train_src( src_encoder,
                                                 src_classifier,
                                                 source_loader, gpu_flag = True)



    # if not (src_encoder.restored and src_classifier.restored and
    #         params.src_model_trained):
    #     src_encoder, src_classifier = train_src( src_encoder,
    #                                              src_classifier,
    #                                              source_loader, gpu_flag = True)

    # eval source model
    # print("=== Evaluating classifier for source domain ===")
    # eval_src(src_encoder, src_classifier, source_loader, gpu_flag = True)
    # print("=== Evaluating target encoder for source domain ===")
    # eval_src(tgt_encoder, src_classifier, source_loader, gpu_flag = True)

   
       
    # train target encoder by GAN
    # print("=== Training encoder for target domain ===")
    # print(">>> Target Encoder <<<")
    # print(tgt_encoder)
    # print(">>> Critic <<<")
    # print(critic)
    tgt_encoder.load_state_dict(src_encoder.state_dict())


    if(os.path.exists(params.tgt_encoder_restore)):
        tgt_encoder.load_state_dict(torch.load(params.tgt_encoder_restore))

    else:
        tgt_encoder = train_target(src_encoder, tgt_encoder, critic,
                                source_loader, target_loader, gpu_flag = True)


    
    # eval target encoder on test set of target dataset
    print("=== Evaluating classifier for encoded target domain ===")
    print(">>> source only <<<")
    eval_tgt(src_encoder, src_classifier, target_loader, gpu_flag = True)
    print(">>> domain adaption <<<")
    eval_tgt(tgt_encoder, src_classifier, target_loader, gpu_flag = True)


