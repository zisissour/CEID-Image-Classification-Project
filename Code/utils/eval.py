import torch
import numpy as np
import pickle

def nn_eval(model, val_loader, loss_fn):
    """
    Evaluate a PyTorch neural network model on a validation set.

    Parameters:
    - model: The model to be evaluated.
    - val_loader: A PyTorch DataLoader containing the validation set.
    - loss_fn: The loss function to be used for evaluation.

    The accuracy and loss of the model on the validation set will be printed, and
    the labels and predictions will be saved in 'output/total_labels.pkl' and
    'output/total_pred.pkl' respectively.
    """
    model.eval()

    val_loss=0
    correct=0
    
    total_labels=[]
    total_pred=[]

    with torch.no_grad():
        for data, labels in val_loader:

            output = model(data)
            val_loss += loss_fn(output, labels).item()

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()

            total_pred.append(pred)
            total_labels.append(labels)
        
    with open('output/total_labels.pkl', 'wb') as f:
        pickle.dump(np.array(total_labels).flatten(), f)
    with open('output/total_pred.pkl', 'wb') as f:
        pickle.dump(np.array(total_pred).flatten(), f)


    val_loss /= len(val_loader.dataset)
    val_acc = correct / len(val_loader.dataset)

    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')


