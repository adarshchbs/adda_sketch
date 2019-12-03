from torch import nn
from torch import optim
import torch

import params
from utils import make_variable, save_model
from preprocess import preprocess_image

def train_src( source_encoder, source_classifier, data_loader, gpu_flag = False ):

    source_classifier.train()

    optimizer = optim.Adam(
                            list(source_classifier.parameters()),
                            lr = params.c_learning_rate,
                            betas = ( params.beta1, params.beta2 )
                            )
    
    criterion = nn.CrossEntropyLoss()

    for epoch in range( params.num_epochs_classifier ):

        for step, ( images, labels ) in enumerate( data_loader.image_gen('train') ):

            images = preprocess_image( array = images,
                                       split_type = 'train',
                                       use_gpu = gpu_flag  )

            labels = torch.tensor(labels,dtype=torch.long)

            if(gpu_flag == True):
                labels = labels.cuda()


            optimizer.zero_grad()
            feature_encoded = source_encoder( images )
            feature_encoded = feature_encoded.view(data_loader.batch_size,512)
            preds = source_classifier( feature_encoded )
            loss = criterion( preds, labels )

            loss.backward()
            optimizer.step()

            # print step info
            if ((step + 1) % params.log_step_pre == 0):
                print("Epoch [{}/{}] Step [{}/{}]: loss={}"
                      .format(epoch + 1,
                              params.num_epochs_pre,
                              step + 1,
                              int(data_loader.size['train']/data_loader.batch_size),
                              loss.data.item()))

        # eval model on test set
        if ((epoch + 1) % params.eval_step_pre == 0):
            eval_src(source_encoder, source_classifier, data_loader)

        # save model parameters
        if ((epoch + 1) % params.save_step_pre == 0):
            save_model(source_encoder, "ADDA-source-encoder-{}.pt".format(epoch + 1))
            save_model(
                source_classifier, "ADDA-source-classifier-{}.pt".format(epoch + 1))

    source_encoder.train()
    optimizer = optim.Adam(
                            list(source_encoder.parameters()) + list(source_classifier.parameters()),
                            lr = params.c_learning_rate,
                            betas = ( params.beta1, params.beta2 )
                            )
    


    for epoch in range( params.num_epochs_encoder ):

        for step, ( images, labels ) in enumerate( data_loader.image_gen('train') ):

            images = preprocess_image( array = images,
                                       split_type = 'train',
                                       use_gpu = gpu_flag  )

            labels = torch.tensor(labels,dtype=torch.long)

            if(gpu_flag == True):
                labels = labels.cuda()


            optimizer.zero_grad()

            preds = source_classifier( source_encoder( images ) )
            loss = criterion( preds, labels )

            loss.backward()
            optimizer.step()

            # print step info
            if ((step + 1) % params.log_step_pre == 0):
                print("Epoch [{}/{}] Step [{}/{}]: loss={}"
                      .format(epoch + 1,
                              params.num_epochs_pre,
                              step + 1,
                              int(data_loader.size['train']/data_loader.batch_size),
                              loss.data.item()))

        # eval model on test set
        if ((epoch + 1) % params.eval_step_pre == 0):
            eval_src(source_encoder, source_classifier, data_loader)

        # save model parameters
        if ((epoch + 1) % params.save_step_pre == 0):
            save_model(source_encoder, "ADDA-source-encoder-{}.pt".format(epoch + 1))
            save_model(
                source_classifier, "ADDA-source-classifier-{}.pt".format(epoch + 1))


    # # save final model
    save_model(source_encoder, "ADDA-source-encoder-final.pt")
    save_model(source_classifier, "ADDA-source-classifier-final.pt")

    return source_encoder, source_classifier



def eval_src( source_encoder, source_classifier, data_loader, gpu_flag = False ):

    loss = 0
    accuracy = 0 

    source_encoder.eval()
    source_classifier.eval()

    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0

    for (images, labels) in data_loader.image_gen(split_type='val'):
        images = preprocess_image( array = images,
                                       split_type = 'val',
                                       use_gpu = gpu_flag  )

        labels = torch.tensor(labels,dtype=torch.long)

        if(gpu_flag == True):
            lables = lables.cuda()
        preds = source_classifier( source_encoder( images ) )
        loss += criterion( preds, labels ).item()

        _, predicted = torch.max(preds.data,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # pred_cls = preds.data.max(1)[1]
        # print(pred_cls.eq(labels.data).cpu().sum())
        # accuracy += pred_cls.eq(labels.data).cpu().sum() / len(labels)

    
    loss /= data_loader.size['val']
    # accuracy /= len( data_loader )
    accuracy = correct/total

    print("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, accuracy))

