# Make imports
import predict_helper
import argparse

# define Mandatory and Optional Arguments for the script
parser = argparse.ArgumentParser(description="Neural Network Settings for predictio ")
parser.add_argument('image', type=str)
parser.add_argument('--checkpoint', type=str, default='./checkpoint.pth')
parser.add_argument('--top_k', type=int, default=5)
parser.add_argument('--category_names', type=str, default='cat_to_name.json')
parser.add_argument('--gpu', action="store_true", default=True)

# setting values data loading
args = parser.parse_args()
    
# Load the model
model = predict_helper.load_checkpoint(args.checkpoint)

#Predict the image
predict_helper.image_predict(args.image, model, class_name_dic=args.category_names, topk=args.top_k, gpu=args.gpu)
