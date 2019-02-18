import argparse

#parser = argparse.ArgumentParser() 

def get_input_args():
    parser = argparse.ArgumentParser()     
    #Mandatory Input
    parser.add_argument('dir', type = str, help = 'path to the folder of flower images', default = None)    
    #Optional Input
    parser.add_argument('--save_dir', type = str ,  default = 'checkpoint.pth', help = 'path to the folder checkpoint')
    parser.add_argument('--arch', type = str, default = 'vgg16')
    parser.add_argument('--learning_rate', type = float, default = 0.001)
    parser.add_argument('--hidden_units', type = int, default = 512)
    parser.add_argument('--epochs', type = int, default = 2)
    parser.add_argument('--cat_to_name', type = str, default = 'cat_to_name.json')
    parser.add_argument('--gpu', type = str,  default = 'cuda' )

    return parser.parse_args()

def get_input_args_predict():
    parser = argparse.ArgumentParser() 
    #Mandatory
    parser = argparse.ArgumentParser() 
    parser.add_argument('image_path', type = str, help = "path to the folder of flower images e.g.flowers/test/1/image_06743.jpg", default = None)
    
    parser.add_argument('checkpoint', type = str, help='Enter Path to checkpoint, checkpoint.pth', default = None)

    #Optional
    parser.add_argument('--topk', type = int, default = 5 )
    parser.add_argument('--cat_to_name', type = str, default = 'cat_to_name.json')
    parser.add_argument('--gpu', type = str, default = 'cpu')
    parser.add_argument('--arch', type = str, default = 'vgg16')


    return parser.parse_args()

    
    
    