# Image-Classifier

### Project Overview
- Load and preprocess a dataset of fowers images
- Train the image classifier on your dataset of flowers
- Use the trained classifier to predict image content
- Create a command line application that can trained on any set of labeled images, and make predictions

### Data
- The data used specifically for this assignment are a flower database(.json file). It is not provided in the repository as it's larger than what github allows.
- The data can be downloaded [here](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz)
- Extract the folder and name the folder containing all data "flowers" for the jupyter notebook to work properly

- The data need to comprised of 3 folders: test, train, validate

- Inside the train, test and validate folders there should be folders bearing a specific number which corresponds to a specific category, clarified in the json file
- For example, if we have the image x.jpg and it is a lotus it could be in a path like this /test/5/x.jpg and json file would be like this {...5:"lotus",...}

### Install(run on command line)
- pip install pandas
- pip install matplotlib
- pip install Pillow
- pip install requests
- pip install notebook
- In order to install Pytorch follow the instructions given on the [official site](https://pytorch.org/)

### Running the command line application
- Train a new network on a data set with **train.py**
  - Basic Usage : ```python train.py data_directory```<br/>
  - Prints out current epoch, training loss, validation loss, and validation accuracy as the netowrk trains
  - Options:
    - Set direcotry to save checkpoints: ```python train.py data_dor --save_dir save_directory```
    - Choose arcitecture (densenet121 or vgg16 available): ```python train.py data_dir --arch "vgg16"```
    - Set hyperparameters: ```python train.py data_dir --learning_rate 0.001 --hidden_layer1 120 --epochs 20```
    - Use GPU for training: ```python train.py data_dir --gpu gpu```
  - Output: A trained network ready with checkpoint saved for doing parsing of flower images and identifying the species.
    
- Predict flower name from an image with **predict.py** along with the probability of that name. That is you'll pass in a single image /path/to/image and return the flower name and class probability
  - Basic usage: ```python predict.py /path/to/image checkpoint```
  - Options:
    - Return top K most likely classes: ```python predict.py input checkpoint ---top_k 3```
    - Use a mapping of categories to real names: ```python predict.py input checkpoint --category_names cat_To_name.json```
    - Use GPU for inference: ```python predict.py input checkpoint --gpu```

### Running the ipynb
- Use jupyter notebook

