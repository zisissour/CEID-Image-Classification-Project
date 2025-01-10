import torch
import torch.utils
import torchvision
from src.models import CNNClassifier, GenericClassifer, Autoencoder
from utils.train import nn_train
from utils.feature_extractors import brief_fe, pca_fe, ae_fe, fd_fe, hog_fe
from utils.eval import nn_eval
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='cnn')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--data_percentage', type=float, default=1)
parser.add_argument('--train', action='store_true')
args = parser.parse_args()



# Load the MNIST dataset
dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

train_dataset, _ = torch.utils.data.random_split(train_dataset, [args.data_percentage,1-args.data_percentage])

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=256,
    shuffle=True
)

test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=256,
    shuffle=True
)

val_dataloader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        root='./data',
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True
    ),
    batch_size=1,
    shuffle=True
)

if(args.model=="cnn"):


    if(args.train):
        model = CNNClassifier()

        opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        loss_fn = torch.nn.CrossEntropyLoss()

        nn_train(model, train_dataloader, test_dataloader, opt, loss_fn, args.epochs)

    model = CNNClassifier()
    model.load_state_dict(torch.load('output/model.pt'))

    nn_eval(model, val_dataloader, loss_fn)

elif(args.model=="hog"):

    if (args.train):
        print("Extracting features...")

        train_features = hog_fe(train_dataloader)
        test_features = hog_fe(test_dataloader)

        train_features_dataloader = torch.utils.data.DataLoader(train_features, batch_size=256, shuffle=True)
        test_features_dataloader = torch.utils.data.DataLoader(test_features, batch_size=256, shuffle=True)

        print("Features extracted. Starting model training...")

        model = GenericClassifer(81)

        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = torch.nn.CrossEntropyLoss()

        nn_train(model, train_features_dataloader, test_features_dataloader, opt, loss_fn, args.epochs)
    
    model = GenericClassifer(81)
    model.load_state_dict(torch.load('output/model.pt'))
    loss_fn = torch.nn.CrossEntropyLoss()

    print("Starting evaluation data feature extraction...")
    val_features = hog_fe(val_dataloader)
    val_features_dataloader = torch.utils.data.DataLoader(val_features, batch_size=1, shuffle=True)

    print("Starting evaluation...")
    nn_eval(model, val_features_dataloader, loss_fn)

elif (args.model=="brief"):

    if(args.train):      

        print("Starting feature extraction...")

        train_features = brief_fe(train_dataloader)

        with open('output/brief_gmm.pkl', 'rb') as f:
            gmm = pickle.load(f)
            
        test_features = brief_fe(test_dataloader, gmm)
            

        train_features_dataloader = torch.utils.data.DataLoader(train_features, batch_size=256, shuffle=True)
        test_features_dataloader = torch.utils.data.DataLoader(test_features, batch_size=256, shuffle=True)
        

        print("Feature extraction completed. Starting model training...")

        model = GenericClassifer(4*128)

        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = torch.nn.CrossEntropyLoss()

        nn_train(model, train_features_dataloader, test_features_dataloader, opt, loss_fn, args.epochs)
    

    model = GenericClassifer(4*128)
    model.load_state_dict(torch.load('output/model.pt'))
    loss_fn = torch.nn.CrossEntropyLoss()

    with open('output/brief_gmm.pkl', 'rb') as f:
            gmm = pickle.load(f)

    print("Starting evaluation data feature extraction...")

    val_features = brief_fe(val_dataloader, gmm)
    val_features_dataloader = torch.utils.data.DataLoader(val_features, batch_size=1, shuffle=True)

    print("Evaluation data feature extraction completed. Starting evaluation...")
    nn_eval(model, val_features_dataloader, loss_fn)

elif(args.model=="pca"):
    if(args.train):
        print("Starting feature extraction...")

        train_features = pca_fe(train_dataloader)
          
        with open("output/pca.pkl", "rb") as f:
            pca = pickle.load(f)

        test_features = pca_fe(test_dataloader,pca)

        train_features_dataloader = torch.utils.data.DataLoader(train_features, batch_size=256, shuffle=True)
        test_features_dataloader = torch.utils.data.DataLoader(test_features, batch_size=256, shuffle=True)
        

        print("Feature extraction completed. Starting model training...")

        model = GenericClassifer(128)

        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = torch.nn.CrossEntropyLoss()

        nn_train(model, train_features_dataloader, test_features_dataloader, opt, loss_fn, args.epochs)
    
    model = GenericClassifer(128)
    model.load_state_dict(torch.load('output/model.pt'))
    loss_fn = torch.nn.CrossEntropyLoss()

    with open('output/pca.pkl', 'rb') as f:
            gmm = pickle.load(f)

    print("Starting evaluation data feature extraction...")

    val_features = pca_fe(val_dataloader, pca)
    val_features_dataloader = torch.utils.data.DataLoader(val_features, batch_size=1, shuffle=True)

    print("Evaluation data feature extraction completed. Starting evaluation...")
    nn_eval(model, val_features_dataloader, loss_fn)

elif(args.model=="ae"):
    if(args.train):
        print("Starting feature extraction...")


        train_features, test_features = ae_fe(train_dataloader, test_dataloader)
        train_features_dataloader = torch.utils.data.DataLoader(train_features, batch_size=256, shuffle=True)
        test_features_dataloader = torch.utils.data.DataLoader(test_features, batch_size=256, shuffle=True)

        print("Feature extraction completed. Starting model training...")

        model = GenericClassifer(128)

        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = torch.nn.CrossEntropyLoss()

        nn_train(model, train_features_dataloader, test_features_dataloader, opt, loss_fn, args.epochs)
    
    model = GenericClassifer(128)
    model.load_state_dict(torch.load('output/model.pt'))
    loss_fn = torch.nn.CrossEntropyLoss()

    ae = Autoencoder()
    ae.load_state_dict(torch.load('output/autoencoder.pt'))

    print("Starting evaluation data feature extraction...")

    val_features = ae_fe(None,val_dataloader,ae)
    val_features_dataloader = torch.utils.data.DataLoader(val_features, batch_size=1, shuffle=True)

    print("Evaluation data feature extraction completed. Starting evaluation...")
    nn_eval(model, val_features_dataloader, loss_fn)

elif(args.model=="vit"):
    if (args.train):
        print("Loading features...")

        with open("data/vit_train_features", "rb") as f:
            train_features = torch.load(f) 
        
        with open("data/vit_test_features", "rb") as f:
            test_features = torch.load(f)
        
        train_features, _ = torch.utils.data.random_split(train_features, [args.data_percentage,1-args.data_percentage])

        train_features_dataloader = torch.utils.data.DataLoader(train_features, batch_size=256, shuffle=True)
        test_features_dataloader = torch.utils.data.DataLoader(test_features, batch_size=256, shuffle=True)

        print("Features loaded. Starting model training...")

        model = GenericClassifer(384)

        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = torch.nn.CrossEntropyLoss()

        nn_train(model, train_features_dataloader, test_features_dataloader, opt, loss_fn, args.epochs)
    
    model = GenericClassifer(384)
    model.load_state_dict(torch.load('output/model.pt'))
    loss_fn = torch.nn.CrossEntropyLoss()

    with open("data/vit_test_features", "rb") as f:
            val_features = torch.load(f)

    val_features_dataloader = torch.utils.data.DataLoader(val_features, batch_size=1, shuffle=True)

    print("Starting evaluation...")
    nn_eval(model, val_features_dataloader, loss_fn)

elif(args.model=="fd"):
    if (args.train):
        print("Extracting features...")

        train_features = fd_fe(train_dataloader)
        test_features = fd_fe(test_dataloader)

        train_features_dataloader = torch.utils.data.DataLoader(train_features, batch_size=256, shuffle=True)
        test_features_dataloader = torch.utils.data.DataLoader(test_features, batch_size=256, shuffle=True)

        print("Features extracted. Starting model training...")

        model = GenericClassifer(200)

        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = torch.nn.CrossEntropyLoss()

        nn_train(model, train_features_dataloader, test_features_dataloader, opt, loss_fn, args.epochs)
    
    model = GenericClassifer(200)
    model.load_state_dict(torch.load('output/model.pt'))
    loss_fn = torch.nn.CrossEntropyLoss()

    print("Starting evaluation data feature extraction...")
    val_features = fd_fe(val_dataloader)
    val_features_dataloader = torch.utils.data.DataLoader(val_features, batch_size=1, shuffle=True)

    print("Starting evaluation...")
    nn_eval(model, val_features_dataloader, loss_fn)
