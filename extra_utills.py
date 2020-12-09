# Imports here
import torch
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
 
# converts a PyTorch tensor and displays it
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


# Do validation
def model_accuracy(model,testloader):
    accuracy = 0
  
    model.eval() # switching to evaluation mode so that dropout is turned off
   
    # Turn off gradients for validation, saves memory and computations
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
         
            # Calculate Validation loss and Validation accuracy for test dat set
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Test accuracy: {accuracy/len(testloader):.3f}")

    model.train() #Set model back to training mode
    
# Display an image along with the top 5 classes
def image_predict_visual(image_path, model, class_name_dic = 'cat_to_name.json'):
 
    # Predict the top K classses using the predict function
    probs,class_indexes,class_names = predict(image_path, model, class_name_dic)
   
    # Show the image
    ax = imshow(process_image(image_path))

    # Set title as the actual flower name
    flower_number = image_path.split('/')[2]
    ax.set_title(class_to_name[flower_number])

    # Plot the graph
    plt.figure(figsize=(5,5))
    plt.barh(range(len(probs)),probs)
    plt.yticks(range(len(probs)),class_names)
    plt.show()
    
def load_checkpoint_retrain(path):
    checkpoint = torch.load(path)

    #load the model
    model = checkpoint['model']
    
    #load the state_dicts
    model.load_state_dict(checkpoint['state_dict'])
    
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    
    model.idx_to_class = checkpoint['idx_to_class']
    
    return model
