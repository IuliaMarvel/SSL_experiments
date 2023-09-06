from mmpretrain import get_model

model_to_weights = dict()
model_to_weights['SimCLR'] = 'simclr_resnet50_16xb256-coslr-200e_in1k'
model_to_weights['BarlowTwins'] = 'barlowtwins_resnet50_8xb256-coslr-300e_in1k'
model_to_weights['Dino'] = 'vit-small-p14_dinov2-pre_3rdparty'

def get_ssl_model(model_name):
    return get_model(model_to_weights[model_name], pretrained=True)
