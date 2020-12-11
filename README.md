# Image-Classifier

### Project Overview

### Data
- The data used specifically for this assignment are a flower database(.json file). It is not provided in the repository as it's larger than what github allows.
- The data can be found in https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz
- Extract the folder and name the folder containing all data "flowers" for the jupyter notebook to work properly

- The data need to comprised of 3 folders: test, train, validate

- Inside the train, test and validate folders there should be folders bearing a specific number which corresponds to a specific category, clarified in the json file
- For example, if we have the image x.jpg and it is a lotus it could be in a path like this /test/5/x.jpg and json file would be like this {...5:"lotus",...}

### Install(run on command line)
- pip install pandas
- pip install torch===1.7.0 torchvision===0.8.1 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html (might differ based on the PC used. Should refer to the pytorvh website for better instructions)
- pip install matplotlib
- pip install Pillow
- pip install requests
- pip install notebook

### Run(command line application)


### Run(jupyter notebook)
Go to command line for the and enter:
- jupyter notebook

