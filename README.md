# Image Classifier 



## Code predict.py


### Load libraries

    import torch
    from torchvision import datasets, transforms, models
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt
    import json
    import get_input_args as args
    from torch import nn
    from collections import OrderedDict
    #import train 

### Load command line arguments
    #Load predict command line args
    arg = args.get_input_args_predict()
    image_path = arg.image_path
    checkpoint = arg.checkpoint #Name checkpoint file, e.g. checkpoint.pth
    topk = arg.topk
    cat_to_name = arg.cat_to_name
    gpu = arg.gpu
    arch = arg.arch
    
    
### Create library and mapps integer encoded categories to the actual names of the flowers
    #Gives a dictionary, mapping the integer encoded categories to the actual names of the flowers. Used in predict()
    with open(cat_to_name, 'r') as f:
        cat_to_name = json.load(f)
    cat_to_name

### Create classifier
    def build_network(input_size, hidden_units, model, output_size = 102, drpout = 0.2, output_features = 1000):   
            for param in model.parameters():
                param.requires_grad = False

            classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_size, output_features)), 
                                                    ('relu', nn.ReLU()),
                                                    ('dropout', nn.Dropout(p = drpout)),
                                                    ('hidden_units', nn.Linear(output_features, hidden_units)),
                                                    ('fc2', nn.Linear(hidden_units, output_size)),
                                                    ('output', nn.LogSoftmax(dim=1))]))
            model.classifier = classifier
            return model

### Process test image
    #Preprocess test image. #image param = url to testimage
    def process_image(image):
        image = Image.open(image)

        #Resize and crop
        width, height = image.size
        size = 256, 256 #width, height

        if width > height:
            ratio = int(width)/int(height)
            new_width = ratio * size[1]
            image = image.resize((int(new_width), size[1]), Image.ANTIALIAS)
        else:
            ## Calculate for the other case
            ratio = int(height)/int(width)
            new_height = ratio * size[0]
            image = image.resize((size[0], int(new_height)), Image.ANTIALIAS)

        #pil_image.crop((256 - 224) / 2., (256 - 224) / 2., (256 + 224) / 2., (256 + 224) / 2.)
        size = image.size #gets width and height from resized image

        image = image.crop((
            size[0] // 2 - (224/2), #left
            size[1] // 2 - (224/2), #top
            size[0] // 2 + (224/2), #right
            size[1] // 2 + (224/2) # bottom
        ))

        #image = image.crop((0,0,224,224))
        #Turn image into numpy array and normalize
        means = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = np.array(image)
        image = image/255
        image = (image - means) / std

        #Switch color to first dimension in numpy array
        image = image.transpose(2,0,1)

        return image
    
### Display test image 
    #Show test image
    def imshow(image, ax=None, title=None):
        if ax is None:
            fig, ax = plt.subplots()

        # PyTorch tensors assume the color channel is the first dimension
        # but matplotlib assumes is the third dimension
        image = image.transpose((1,2,0))

        #Undo preprocessing
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean

        image = np.clip(image, 0, 1)

        ax.imshow(image)
        return image
    
#image = process_image('flowers/test/1/image_06743.jpg')
#imshow(image)  
#Load Checkpoint

### Load checkpoint
    def load_checkpoint(filepath):
        pretrained_models = {'resnet18': models.resnet18(pretrained=True),
                             'alexnet': models.alexnet(pretrained=True),
                             'squeezenet': models.squeezenet1_0(pretrained=True),
                             'vgg16': models.vgg16(pretrained=True),
                             'densenet': models.densenet121(pretrained=True),
                             'inception': models.inception_v3(pretrained=True)}

        checkpoint = torch.load(filepath)
        #model = pretrained_models[arch]
        input_size = checkpoint['input_size']
        output_size = checkpoint['output_size']
        hidden_units = 512 #checkpoint['hidden_units']
        drpout = 0.2
        pretrained_model = pretrained_models[checkpoint['pretrained_model']]
        #print('Pretrained model', pretrained_model)

        model = build_network(input_size, hidden_units, pretrained_model, output_size, drpout)
        #model.classifier = checkpoint['classifier']
        model.load_state_dict(checkpoint['state_dict'])
        #print("Trained model")
        #print('Trained model',model)
        return model

### Predict five top classes and their probabilties of the test image
    #Predict 5 top classes and probabilities of image
    def predict_image(image_path, model,  checkpoint, topk = 5):    
        model.to(gpu) #model moved to cpu by default
        model.eval()

        #Process PIL image and turn into tensor
        image = process_image(image_path)
        dim_up = None

        if gpu == 'cpu':
            dim_up = torch.tensor(np.array([image])).to(torch.float)
            #image = np.array([image])
            #dim_up = torch.cuda.FloatTensor(image)
        elif gpu == 'cuda':
            image = np.array([image])
            dim_up = torch.cuda.FloatTensor(image)

        with torch.no_grad():
            output = model.forward(dim_up)
        ps = torch.exp(output)

        #x.topk(k) returns highest k probabilities and the indices of those probabilities corresponding to  the classes. 
        You need to convert from these indices to the actual class labels using class_to_idx 
        probs, classes = ps.topk(topk)
        probs = probs[0].to(gpu)
        probs = probs.numpy()

        class_indeces = classes[0].to(gpu)
        class_indeces = class_indeces.numpy()
        class_name = np.array([])

        checkpoint = torch.load(checkpoint)
        for key, items in checkpoint['class_to_idx'].items():
            #print('Class to index pair:', key, items)
            if items in class_indeces:        
                class_name = np.append(class_name, cat_to_name[key])

        return probs, classes, class_name


### Call functions to build model and print out rpredictions
model = load_checkpoint(checkpoint)
probs, classes, class_name = predict_image(image_path , model, checkpoint, topk)
for class_name, probs in list(zip(class_name,probs)):
    print('Flower name: {:15s}  Class prob: {:.3f} %'.format(class_name.capitalize(), probs*100))
