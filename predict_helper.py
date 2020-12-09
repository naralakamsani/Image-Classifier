# Imports here
import torch
from PIL import Image
import numpy as np
import pandas as pd
import json
from torchvision import models


# Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(path):
    checkpoint = torch.load(path)

    # load the model
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    #load classifier
    model.classifier = checkpoint['classifier']
    
    #load the state_dict
    model.load_state_dict(checkpoint['state_dict'])
    
    #load idx to lass dict
    model.idx_to_class = checkpoint['idx_to_class']
    
    return model

def process_image(image):
    
    image = Image.open(image)
    
    # Process a PIL image for use in a PyTorch model
    
    # Get original dimensions
    w, h = image.size

    # Find longer side to crop 
    if w < h:
        resize_size=(256,256**2)
    else: 
        resize_size=(256**2, 256)
 
    image.thumbnail(resize_size)
    
    # crop
    crop_size = 224
    width, height = image.size   # Get dimensions
    
    left = (width - crop_size)/2
    top = (height - crop_size)/2
    right = (width + crop_size)/2
    bottom = (height + crop_size)/2
    
    image = image.crop((left, top, right, bottom))
    
    np_image = np.array(image)
    
    # normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_array = np.array(image) / 255
    
    image = (image_array - mean) / std
    
    # reorder dimensions for color channel
    image = image.transpose((2,0,1))
    
    return torch.from_numpy(image)

# This method should take a path to an image and a model checkpoint, then return the probabilities and classes.
def predict(image_path, model, class_name_dic, topk, gpu=True):
    ''' Predict the class (or classes) of an image using a trained deep learning model. '''
    
    # Implement the code to predict the class from an image file
    
    # Use gpu depending on users input
    if gpu:
        device = 'cuda:0'
    else:
        device = 'cpu'
        
    model.to(device)
    
    # Open the json file and load it.
    with open(class_name_dic, 'r') as f:
        class_to_name = json.load(f)
    
    # loading image and processing it using above defined function
    image = process_image(image_path)
    
    # used to make size of torch as expected. as forward method is working with batches,
    # doing that we will have batch size equal to 1
    image = image.unsqueeze_(0)
    
    # we cannot pass image to model.forward 'as is' as it is expecting tensor, not numpy array
    # converting to tensor
    image = image.to(device).float()
    
    
    model.eval() # turn off gradients for validation, saves memory and computations
    
    with torch.no_grad():
        logps = model.forward(image)
        ps = torch.exp(logps) # converting into a probability
        top_p, top_class = ps.topk(topk)
        
    model.train() # set model back to training mode
    
    # converting idexes to class numbers
    class_nums = []
    for indexes in top_class.cpu().numpy()[0]:
        class_nums.append(model.idx_to_class[indexes])
    
    # coverting class numbers to names
    class_names = []
    for nums in class_nums:
        class_names.append(class_to_name[str(nums)])
    
    probs = top_p.tolist()[0] # converting to list
    
    return probs,class_nums,class_names     

# Display the predicted top 5 classes for particular flower
def image_predict(image_path, model, class_name_dic = 'cat_to_name.json', topk=5, gpu=True):
    
    # Predict the top K classses using the predict function
    probs,class_indexes,class_names = predict(image_path, model, class_name_dic, topk, gpu=gpu)
    
    # Create a pandas dataframe joining class to flower names
    top_preds = pd.DataFrame({'rank':[i for i in range(1,len(probs)+1)],'probability':probs,'flower_name':class_names})
    top_preds = top_preds.set_index('rank')
    
    # Print the top predictions
    print("Below are the top predicted classes and their probabilites for the given image")
    print(top_preds)
    print("There is " + str(probs[0]*100) + "% probability that this flower in the provided image is " + class_names[0])
