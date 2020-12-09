# Make imports
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

# Function to process the data
def process_data(data_dir):
    
    #Defining the directories for the data
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    test_val_transforms = transforms.Compose([transforms.Resize(225),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])


    # Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_dataset = datasets.ImageFolder(valid_dir, transform=test_val_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_val_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64,shuffle=True)
    validloader = torch.utils.data.DataLoader(validation_dataset, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64)

    return trainloader, validloader, testloader, train_dataset.class_to_idx

# Function to create the model
def create_model(arch='vgg16', input_units=25088, hidden_units=512, output_units=102):
    
    # Build and train your network
    model = getattr(models, arch)(pretrained=True)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
        
    # Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
    classifier = nn.Sequential(nn.Linear(input_units,hidden_units),
                              nn.ReLU(),
                              nn.Linear(hidden_units,256),
                              nn.ReLU(),
                              nn.Linear(256,output_units),
                              nn.LogSoftmax(dim=1))

    model.classifier = classifier

    return model

# Train the classifier layers using backpropagation using the pre-trained network to get the features.
def train(model, trainloader, validloader, print_every=5, criterion=nn.NLLLoss(), lr=0.001, epochs=1, gpu=True):

    # Select if cpu or cuda should be used
    if gpu:
        device = 'cuda:0'
    else:
        device = 'cpu'

    model.to(device)
    
    #Define optimizer
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    
    # Define deep learning method
    steps = 0
    running_loss = 0

    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad() # where optimizer is working on classifier paramters only

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() # loss.item () returns scalar value of Loss function

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval() # switching to evaluation mode so that dropout is turned off
                
                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # # Calculate validation loss and validation accuracy for Validation datset
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                #Print the details during the model training
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {test_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validloader):.3f}")
                
                running_loss = 0
                model.train() #Set model back to training mode
          
    return model

#Save the checkpoint
def save_model(model,class_to_idx, arch, save_loc='checkpoint.pth'):
    #saving mapping between predicted class and class name
    #second variable is a class name in numeric
    model.idx_to_class = {v: k for k, v in class_to_idx.items()}

    #creating dictionary
    checkpoint = {
                'arch': arch,
                'classifier': model.cpu().classifier,
                'state_dict': model.state_dict(),
                'idx_to_class': model.idx_to_class
                }

    torch.save (checkpoint, save_loc)




