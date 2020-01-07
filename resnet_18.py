import torch
from torch import nn
import torch.nn.functional as F 



class ResNetEncoder(nn.Module):
    def __init__(self, encoder):
        super( ResNetEncoder, self ).__init__()
        # self.restored = False
        self.encoder = encoder
    
    def forward(self, inputs):
        out = self.encoder(inputs)
        out = out.view(out.size()[0],out.size()[1])
        return out


class ResNetClassifier(nn.Module):
    def __init__(self):
        super( ResNetClassifier, self ).__init__()
        self.fc2 = nn.Linear( in_features = 512, out_features = 87, bias= True )

    def forward(self, inputs):
        out = self.fc2( inputs )
        return out


# a = ResNetEncoder()
# print(a.state_dict()['encoder.7.1.conv2.weight'][0][0:6])

# import numpy as np
# from image_loader import image_loader
# from preprocess import preprocess_image
# path_sketchy = '/home/iacv/project/adda_sketch/dataset/sketches/'
# path_quickdraw = '/home/iacv/project/adda_sketch/dataset/QuickDraw_sketches_final/'
# path_class_list = '/home/iacv/project/adda_sketch/common_class_list.txt'

# class_list = np.loadtxt(path_class_list,dtype='str')
# # load dataset
# source_loader = image_loader(parent_folder_path = path_sketchy,
#                         folder_list= class_list,
#                         split= [0.8,0.2,0] )
# src_data_loader = source_loader.image_gen(split_type='train')

# for image, label in src_data_loader:
#     image = preprocess_image(image, split_type = 'train', use_gpu= False)
#     pred = resnet(image)
#     _, predicted = torch.max(pred.data,1)
#     print(predicted)
#     print(label)
#     break