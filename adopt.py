import os

import torch 
from torch import nn, optim

import numpy as np
import params
from utils import make_variable
from preprocess import preprocess_image, preprocess_image_1
from test import eval_tgt
from itertools import count

def train_target(source_encoder, target_encoder, critic, source_classifier,
                 source_dataloader, target_dataloader, gpu_flag = False, gpu_name = 'cuda:0'):
    
    target_encoder.train()
    critic.train()
    source_encoder.eval()

    for p in source_encoder.parameters():
        p.requires_grad = False
    criterion = nn.CrossEntropyLoss()
    optimizer_target = optim.Adam(target_encoder.parameters() ,
                               lr=params.c_learning_rate,
                               betas=(params.beta1, params.beta2))
    optimizer_critic = optim.Adam(critic.parameters(),
                                  lr=params.d_learning_rate,
                                  betas=(params.beta1, params.beta2))
    len_data_loader = min(source_dataloader.size['train'], target_dataloader.size['train'] )
    # data_zip = enumerate( zip( source_dataloader, target_dataloader ) )

    for epoch in range(params.num_epochs):
        correct = 0
        total = 0 
        total_loss = 0
        count_ = 0
        for step, (source_image, _), (target_image, _)  in zip(count(),source_dataloader.image_gen(),
                                                                       target_dataloader.image_gen()):
                   
            source_image = preprocess_image( array = source_image,
                                       split_type = 'train',
                                       use_gpu = gpu_flag, gpu_name= gpu_name  )

            target_image = preprocess_image_1( array = target_image,
                                       split_type = 'train',
                                       use_gpu = gpu_flag, gpu_name= gpu_name  )

            optimizer_critic.zero_grad()
            
            source_feature = source_encoder( source_image )
            target_feature = target_encoder( target_image )
            concat_feature = torch.cat( ( source_feature, target_feature ), 0 )

            pred_concat = critic( concat_feature ) 

            source_label = torch.ones( source_feature.size(0) ).long().cuda(gpu_name)
            target_label = torch.zeros( target_feature.size(0) ).long().cuda(gpu_name)
            concat_label = torch.cat( ( source_label, target_label ) )

            loss_critic = criterion( pred_concat, concat_label )
            loss_critic.backward()

            optimizer_critic.step()

            _,pred_cls = torch.max( pred_concat.data, 1 )
            correct += (pred_cls == concat_label).sum().item()
            total += concat_label.size(0)
            total_loss += loss_critic.item()
            count_ += 1
            # acc = correct/concat_label.size(0)
            
            # if(epoch > 99):
            optimizer_critic.zero_grad()
            optimizer_target.zero_grad()

            target_feature = target_encoder( target_image )

            pred_target = critic( target_feature )

            label_target = torch.ones( pred_target.size(0) ).long().cuda(gpu_name)

            loss_target = criterion( pred_target, label_target )
            loss_target.backward()

            optimizer_target.step()

            
            if ((step + 1) % params.log_step == 0):
                print("Epoch [{}/{}] Step [{}/{}]:"
                        "d_loss={:.5f} g_loss={:.5f} acc={:.5f}"
                        .format(epoch + 1,
                                params.num_epochs,
                                step + 1,
                                int(len_data_loader/source_dataloader.batch_size),
                                loss_critic.item(),
                                loss_target.item(),
                                correct/total))
                # print(target_encoder.state_dict()['encoder.7.1.conv2.weight'][0][0:5])
                # print("..........................")

            if ((step + 1) % params.eval_step == 0):
            # torch.save(critic.state_dict(), os.path.join(
            #     params.model_root,
            #     "ADDA-critic-{}.pt".format(epoch + 1)))
            # torch.save(target_encoder.state_dict(), os.path.join(
            #     params.model_root,
            #     "ADDA-target-encoder-{}.pt".format(epoch + 1)))
                print(">>> domain adaption <<<")
                eval_tgt(target_encoder, source_classifier, target_dataloader, gpu_flag = True, gpu_name= gpu_name)


        print( "critic accuracy after {} epochs is {}".format(epoch, correct/total) )
        print( "critic loss {}".format(total_loss/count_) )
        print(pred_cls)

    torch.save(critic.state_dict(), os.path.join(
        params.model_root,
        "ADDA-critic-final.pt"))
    torch.save(target_encoder.state_dict(), os.path.join(
        params.model_root,
        "ADDA-target-encoder-final.pt"))
    
    return target_encoder

