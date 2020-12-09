# Make imports
import train_helper
import argparse

parser = argparse.ArgumentParser(description="Network settings for training")
parser.add_argument('data_dir',type=str)
parser.add_argument('--save_dir', type=str, default='./checkpoint.pth')
parser.add_argument('--arch', type=str, action="store", default="vgg16")
parser.add_argument('--learning_rate', type=int, action="store", default=0.001)
parser.add_argument('--hidden_units', type=int, action="store", default=512)
parser.add_argument('--epochs', type=int, action="store", default=1)
parser.add_argument('--gpu', action="store_true", default=False)
args = parser.parse_args()


trainloader, validloader, testloader, class_to_idx = train_helper.process_data(args.data_dir)

model = train_helper.create_model(arch=args.arch, hidden_units=args.hidden_units)

model = train_helper.train(model, trainloader, validloader, lr=args.learning_rate, epochs=args.epochs, gpu=args.gpu)

train_helper.save_model(model, class_to_idx, args.arch, save_loc=args.save_dir)

