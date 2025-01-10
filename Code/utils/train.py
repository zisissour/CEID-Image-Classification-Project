import torch
import pickle

def nn_train(model, train_loader, test_loader, optimizer, loss_fn, num_epochs):

    """
    Train a PyTorch neural network model.

    Args:
    - model: The model to be trained.
    - train_loader: A PyTorch DataLoader containing the training data.
    - test_loader: A PyTorch DataLoader containing the test data.
    - optimizer: The optimizer to be used for training.
    - loss_fn: The loss function to be used for training.
    - num_epochs: The number of epochs to train the model for.

    The training and test losses will be saved in 'output/train_losses.pkl'
    and 'output/test_losses.pkl' respectively, and the best model will be saved
    in 'output/model.pt'.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    train_losses=[]
    test_losses=[]

    best_loss=9999999
    
    for epoch in range(num_epochs):

        mean_loss = 0

        model.train()
        
        #Feed data to model and update parameters
        for data, labels in train_loader:

            data, labels = data.to(device), labels.to(device)

            output = model(data)
            loss = loss_fn(output, labels)
            mean_loss += loss.item()

            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        mean_loss /= len(train_loader.dataset)
        train_losses.append(mean_loss)

        model.eval()

        test_loss = 0
        correct = 0

        with torch.no_grad():

            #Model evaluation
            for data, labels in test_loader:

                data, labels = data.to(device), labels.to(device)

                output = model(data)
                test_loss += loss_fn(output, labels).item()

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()
        
        test_loss /= len(test_loader.dataset)
        test_acc = correct / len(test_loader.dataset)
        test_losses.append(test_loss)

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {test_loss:.4f}, Acc: {test_acc:.4f}')

        #If the model is the best so far, save it
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), 'output/model.pt')

    with open('output/train_losses.pkl', 'wb') as f:
        pickle.dump(train_losses, f)
    with open('output/test_losses.pkl', 'wb') as f:
        pickle.dump(test_losses, f)

def ae_train(model, train_loader, test_loader, optimizer, loss_fn, num_epochs):
    """
    Train an Autoencoder model.

    Parameters:
    - model: The model to be trained.
    - train_loader: A PyTorch DataLoader containing the training data.
    - test_loader: A PyTorch DataLoader containing the test data.
    - optimizer: The optimizer to be used for training.
    - loss_fn: The loss function to be used for training.
    - num_epochs: The number of epochs to train the model for.

    The training and test losses will be saved in 'output/ae_train_losses.pkl'
    and 'output/ae_test_losses.pkl' respectively, and the best model will be saved
    in 'output/autoencoder.pt'.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    train_losses=[]
    test_losses=[]

    best_loss=9999999
    
    for epoch in range(num_epochs):

        mean_loss = 0

        model.train()
        
        #Feed data to model and update parameters
        for data, _ in train_loader:

            data = data.to(device)

            output = model(data)

            #Calculate loss, the mse between the output and the original data
            loss = loss_fn(output, data)
            mean_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        mean_loss /= len(train_loader.dataset)
        train_losses.append(mean_loss)

        model.eval()

        test_loss = 0

        with torch.no_grad():

            #Model evaluation
            for data, _ in test_loader:

                data = data.to(device)

                output = model(data)
                test_loss += loss_fn(output, data).item()

                pred = output.argmax(dim=1, keepdim=True)
        
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {test_loss:.4f}')

        #If the model is the best so far, save it
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), 'output/autoencoder.pt')

    with open('output/ae_train_losses.pkl', 'wb') as f:
        pickle.dump(train_losses, f)
    with open('output/ae_test_losses.pkl', 'wb') as f:
        pickle.dump(test_losses, f)
