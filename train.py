import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
import torch.nn.functional as F
import get_input_args as args 
import time
from tqdm import tqdm

#Load train command line args
if __name__ == "__main__":
    arg = args.get_input_args()

    data_dir = arg.dir
    learning_rate = arg.learning_rate
    epochs = arg.epochs
    arch = arg.arch
    gpu = arg.gpu
    save_dir = arg.save_dir
    hidden_units = arg.hidden_units
    cat_to_name = arg.cat_to_name
    output_size = 102
    drpout = 0.2


    def load_data(data_dir):
        #data_dir = 'flowers'
        train_dir = data_dir + '/train'
        valid_dir = data_dir + '/valid'
        test_dir = data_dir + '/test'

        #Define transforms for training, validation and testing sets
        train_transforms = transforms.Compose([#transforms.Resize(224),
                                               #transforms.CenterCrop(224), 
                                               transforms.RandomResizedCrop(224, scale = (0.07, 1.0)),
                                               transforms.RandomRotation(30),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229,0.224,0.225])])

        validation_transforms = transforms.Compose([transforms.Resize(256),                
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229,0.224,0.225])])
        test_transforms = transforms.Compose([transforms.Resize(256),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], 
                                                                    [0.229, 0.224, 0.225])])

        #Load the image datasets
        traindata = datasets.ImageFolder(train_dir, transform = train_transforms)
        validationdata = datasets.ImageFolder(valid_dir, transform = validation_transforms)
        testdata = datasets.ImageFolder(test_dir, transform = test_transforms)
        image_datasets = {'train': traindata, 'valid': validationdata, 'test': testdata}

        #Define the dataloader
        trainloader = torch.utils.data.DataLoader(traindata, batch_size = 64, 
                                                  shuffle = True)
        validationloader = torch.utils.data.DataLoader(validationdata, batch_size = 32, shuffle = True)
        testloader = torch.utils.data.DataLoader(testdata, batch_size = 32, shuffle = True)

        return trainloader, validationloader, testloader, image_datasets


    def pretrained_model(arch):
        input_size = 0
        pretrained_models = {'resnet18': models.resnet18(pretrained=True),
                             'alexnet': models.alexnet(pretrained=True),
                             'squeezenet1_0': models.squeezenet1_0(pretrained=True),
                             'vgg16': models.vgg16(pretrained=True),
                             'densenet121': models.densenet121(pretrained=True),
                             'inception_v3': models.inception_v3(pretrained=True)}

        model = pretrained_models[arch]

        if arch == 'vgg16':
            #input_size = int(model.classifier[0].in_features)  
            input_size = model.classifier[0].in_features
        elif arch == 'alexnet':
            input_size = model.classifier[1].in_features
        elif arch == 'squeezenet1_0':
            input_size = 512
        elif arch == 'densenet121':
            input_size = model.classifier.in_features
        elif arch == 'inception_v3':
            input_size = model.model.fc.in_features

        return  model, input_size

    #Build network
    def build_network(input_size, model, output_size = 102, drpout = 0.2, output_features = 1000):   
        for param in model.parameters():
            param.requires_grad = False

        classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_size, output_features)), 
                                                ('relu', nn.ReLU()),
                                                ('dropout', nn.Dropout(p = drpout)),
                                                ('hidden_units',nn.Linear(output_features, hidden_units)),
                                                ('fc2', nn.Linear(hidden_units, output_size)),
                                                ('output', nn.LogSoftmax(dim=1))]))
        model.classifier = classifier
        return model


    def train_network(model, trainloader, validationloader, learning_rate, gpu = 'gpu', epochs = 2): 
        criterion = nn.NLLLoss()
        optimizer = optim.SGD(model.classifier.parameters(), lr = learning_rate)    
        #epochs = epochs
        steps = 0
        every_step = 40
        running_loss = 0

        for e in tqdm(range(epochs)):
            time.sleep(1)
            model.to(gpu) #Move model to gpu for faster processing
            model.train()

            for ii, (images, labels) in enumerate(trainloader):

                steps += 1
                images, labels = images.to(gpu), labels.to(gpu)

                #Do a forward pass
                outputs = model.forward(images)
                loss = criterion(outputs, labels)
                loss.backward() #Backpropagation through the network (criterion part)
                optimizer.step() #update weights and biases of classifier

                running_loss += loss.item()

                ##Run validation
                if steps % every_step == 0:
                    model.eval()
                    test_loss, accuracy = validation(model, validationloader, criterion)
                    print('Epoch: {}/{}' .format(e+1, epochs), 
                          'Training loss: {:.3f}'.format(running_loss / every_step),
                          'Test Loss: {:.3f}'.format(test_loss/len(validationloader)),
                          'Accuracy: {:.3f}'.format(accuracy/len(validationloader)))                                        
                    running_loss = 0
                    model.train()

        return model
                #if ii == 121:
                    #break

    def validation(model, validationloader, criterion):
        test_loss = 0
        accuracy = 0

        with torch.no_grad():
            for ii, (images, labels) in enumerate(validationloader):
                images, labels = images.to(gpu), labels.to(gpu)

                outputs = model.forward(images)
                test_loss += criterion(outputs, labels).item()     
                ps = torch.exp(outputs)
                equality = (labels.data == ps.max(dim=1)[1])
                accuracy += equality.type(torch.FloatTensor).mean()

            return test_loss, accuracy

    def save_checkpoint(model, image_datasets, output_size, input_size):   
        model.class_to_idx = image_datasets['train'].class_to_idx
        checkpoint = {'pretrained_model': arch,
                      'input_size': input_size, 
                      'output_size': output_size,
                      'epochs' : epochs,
                      'class_to_idx': model.class_to_idx,
                      'classifier': model.classifier,
                      'state_dict': model.state_dict()}
        #torch.save(checkpoint, 'checkpoint.pth')
        torch.save(checkpoint, save_dir)
        print('Checkpoint saved')
        
    #Get pretrained model from user input and input size for classifier. VGG16 by default        
    pretrained_model, input_size = pretrained_model(arch) 
    #print(input_size)

    #Build model from pretrained model
    model = build_network(input_size,  pretrained_model, output_size, drpout)

    #Load image data from data_dir provided by user input
    trainloader, validationloader, testloader, image_datasets = load_data(data_dir)

    #Train model
    model = train_network(model, trainloader, validationloader, learning_rate, gpu, epochs)

    #Save model
    save_checkpoint(model, image_datasets, output_size, input_size)




 
            
            
       
            
        
        
    




    






    
