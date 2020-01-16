import os

import torch 
from torch import nn, optim

import numpy as np
import params
from utils import make_variable
import loss_cdan
from preprocess import preprocess_image, preprocess_image_1
from test import eval_tgt
from itertools import count

def train_target( encoder, classifier, critic, random_layer,
                 source_dataloader, target_dataloader, gpu_flag = False, gpu_name = 'cuda:0'):
    
    encoder.train()
    critic.train()

    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)
    optimizer = optim.Adam( list(encoder.parameters()) + list(classifier.parameters()) ,
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
        for step, (source_image, labels_source), (target_image, _)  in zip(count(),source_dataloader.image_gen(),
                                                                       target_dataloader.image_gen()):
                   
            source_image = preprocess_image( array = source_image,
                                       split_type = 'train',
                                       use_gpu = gpu_flag, gpu_name = gpu_name  )

            target_image = preprocess_image_1( array = target_image,
                                       split_type = 'train',
                                       use_gpu = gpu_flag, gpu_name = gpu_name  )
            
            labels_source = torch.tensor(labels_source, dtype=torch.long).cuda(gpu_name)


            optimizer_critic.zero_grad()
            optimizer.zero_grad()

            source_feature = encoder( source_image )
            target_feature = encoder( target_image )
            concat_feature = torch.cat( ( source_feature, target_feature ), 0 )
            
            source_output = classifier( source_feature )
            target_output = classifier( target_feature )
            concat_output = torch.cat( (source_output, target_output), 0 )
            
            softmax_out = softmax(concat_output)

            entropy = loss_cdan.Entropy(softmax_out)
            
            transfer_loss = loss_cdan.CDAN([concat_feature, softmax_out], critic, entropy, random_layer, gpu_name)
            
            

            classifier_loss = nn.CrossEntropyLoss()( source_output, labels_source)

            total_loss =  classifier_loss  -0.1 * transfer_loss
            

            total_loss.backward(retain_graph = True)
            transfer_loss.backward()

            if(epoch > 3):
                optimizer.step()

                optimizer_critic.step()

            _,pred_cls = torch.max( source_output.data, 1 )
            correct += (pred_cls == labels_source).sum().item()
            total += labels_source.size(0)
            
            
            if ((step + 1) % params.log_step == 0):
                print("Epoch [{}/{}] Step [{}/{}]:"
                        "transfer_loss={:.5f} total_loss={:.5f} acc={:.5f}"
                        .format(epoch + 1,
                                params.num_epochs,
                                step + 1,
                                int(len_data_loader/source_dataloader.batch_size),
                                transfer_loss.item(),
                                total_loss.item(),
                                correct/total))
                # print(encoder.state_dict()['encoder.7.1.conv2.weight'][0][0:5])
                # print("..........................")
    
        print( "critic accuracy after {} epochs is {}".format(epoch, correct/total) )
        # print( "critic loss {}".format(total_loss/count_) )
        print(pred_cls)
        #############################
        # 2.4 save model parameters #
        #############################
        if ((epoch + 1) % params.save_step == 0):
            # torch.save(critic.state_dict(), os.path.join(
            #     params.model_root,
            #     "ADDA-critic-{}.pt".format(epoch + 1)))
            # torch.save(encoder.state_dict(), os.path.join(
            #     params.model_root,
            #     "ADDA-target-encoder-{}.pt".format(epoch + 1)))

            print(">>> domain adaption <<<")
            eval_tgt(encoder, classifier, target_dataloader, gpu_flag = True, gpu_name= gpu_name)

    torch.save(critic.state_dict(), os.path.join(
        params.model_root,
        "ADDA-critic-final.pt"))
    # torch.save(random_layer.state_dict(), os.path.join(
    #     params.model_root,
    #     "ADDA-random_layer-final.pt"))
    torch.save(encoder.state_dict(), os.path.join(
        params.model_root,
        "ADDA-target-encoder-final.pt"))
    
    return encoder

