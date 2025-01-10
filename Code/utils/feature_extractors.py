from skimage.feature import BRIEF, corner_harris, corner_peaks, fisher_vector, learn_gmm, hog
import numpy as np
from sklearn.decomposition import PCA
import torch
import pickle
from src.models import Autoencoder
from utils.train import ae_train
import cv2
from skimage.filters import threshold_otsu
from scipy.fft import fft

def brief_fe(dataloader, gmm=None):
   
    """
    Compute the Fisher vectors from the BRIEF descriptors of the images in the dataloader.

    Parameters:
    - dataloader: A PyTorch DataLoader containing the images to be processed.
    - gmm: A Gaussian Mixture Model (GMM) to be used for computing the Fisher vectors. If None a new GMM will be created and trained.

    Returns:
    A PyTorch TensorDataset containing the Fisher vectors and their corresponding labels.
    """
    extractor = BRIEF(patch_size=5, descriptor_size=128)

    descriptors = []
    labels_list = []
    counter=0
    for data, labels in dataloader:
        for image in data:
            # Compute the descriptors for the image
            image = np.array(image).reshape(28, 28)
            keypoints = corner_peaks(corner_harris(image), min_distance=1)[:4]
            
            extractor.extract(image, keypoints)
            features = extractor.descriptors
            # Convert the descriptors to int8 numpy array
            features = np.array(features, dtype=np.float32)
            features = features.flatten()

            if len(features) != 4*128:
                features = np.pad(features, (0, 4*128-len(features)), 'constant')
            
            descriptors.append(features)
        # Append the labels to the list
        labels_list.append(labels)
    if gmm is None:
        print("Training Gaussian Mixture Model...")

        gmm =learn_gmm([descriptors], n_modes=4)

        print("Training complete.")
    
    with open('output/brief_gmm.pkl', 'wb') as f:
        pickle.dump(gmm, f)

    # Compute the fisher vectors for the descriptors
    fisher_vectors = [fisher_vector(x, gmm) for x in descriptors]
    # Convert the fisher vectors to float32 numpy array
    fisher_vectors = np.array(fisher_vectors, dtype=np.float32)
    
    
    return torch.utils.data.TensorDataset(
        torch.from_numpy(np.array(descriptors)),
        torch.cat(labels_list)
    )

def pca_fe(dataloader, pca=None):

    """
    Compute the Principal Component Analysis features from the images in the dataloader.

    Parameters:
    - dataloader: A PyTorch DataLoader containing the images to be processed.
    - pca: A PCA object to be used for computing the features. If None, a new PCA object will be created.

    Returns:
    A PyTorch TensorDataset containing the PCA features and their corresponding labels.
    """
    images_list=[]
    labels_list=[]

    for data, labels in dataloader:
        for image in data:
            image = image.numpy().flatten()
            images_list.append(image)
        
        labels_list.append(labels)
    
    if pca is None:
        pca = PCA(n_components=128)
        pca.fit(images_list)

        with open("output/pca.pkl", "wb") as f:
            pickle.dump(pca, f)
    
    features = pca.transform(images_list)
    
    features = np.array(features, dtype=np.float32)

    return torch.utils.data.TensorDataset(
        torch.from_numpy(features),
        torch.cat(labels_list)
    )

def ae_fe(train_dataloader, test_dataloader, ae=None):

    """
    Compute the Autoencoder features from the images in the dataloader.

    Parameters:
    - train_dataloader: A PyTorch DataLoader containing the images to be processed for training.
    - test_dataloader: A PyTorch DataLoader containing the images to be processed for testing.
    - ae: An Autoencoder object to be used for computing the features. If None, a new Autoencoder object will be created.

    Returns:
    Two PyTorch TensorDataset containing the Autoencoder features and their corresponding labels for training and testing respectively.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if ae is None:
        ae = Autoencoder()

        ae = ae.to(device)
        
        opt = torch.optim.Adam(ae.parameters(), lr=0.001)
        loss_fn = torch.nn.MSELoss()

        print("Training autoencoder...")

        ae_train(ae, train_dataloader,test_dataloader, opt, loss_fn, 5)

        print("Training complete.")

        ae.load_state_dict(torch.load('output/autoencoder.pt'))

        train_features =[]
        train_labels_list =[]

        for data, labels in train_dataloader:
            data, labels = data.to(device), labels.to(device)
            encoded = ae.encoder(data)
            train_features.append(encoded.detach())
            train_labels_list.append(labels.detach())
        
        test_features =[]
        test_labels_list =[]

        for data, labels in test_dataloader:
            data, labels = data.to(device), labels.to(device)

            encoded = ae.encoder(data)
            test_features.append(encoded.detach())
            test_labels_list.append(labels.detach())

        return torch.utils.data.TensorDataset(
            torch.cat(train_features),
            torch.cat(train_labels_list)
        ), torch.utils.data.TensorDataset(
            torch.cat(test_features),
            torch.cat(test_labels_list)
        )
    else:

        ae = ae.to(device)

        test_features =[]
        test_labels_list =[]

        for data, labels in test_dataloader:
            data, labels = data.to(device), labels.to(device)

            encoded = ae.encoder(data)
            test_features.append(encoded)
            test_labels_list.append(labels)

        return torch.utils.data.TensorDataset(
            torch.cat(test_features),
            torch.cat(test_labels_list)
        )

def fd_fe(dataloader):

    """
    Compute the Fourier descriptors from the images in the dataloader.

    Parameters:
    - dataloader: A PyTorch DataLoader containing the images to be processed.

    Returns:
    A PyTorch TensorDataset containing the Fourier descriptors and their corresponding labels.
    """
    features_list=[]
    labels_list=[]

    for data, labels in dataloader:
        for image in data:

            image = image.numpy().reshape(28,28)
            image = image*255
            features = compute_fourier_descriptors(image)
            features_list.append(features)
        
        labels_list.append(labels)
            
    return torch.utils.data.TensorDataset(
        torch.from_numpy(np.array(features_list)),
        torch.cat(labels_list)
    )

def hog_fe(dataloader):

    """
    Compute the HOG features from the images in the dataloader.

    Parameters:
    - dataloader: A PyTorch DataLoader containing the images to be processed.

    Returns:
    A PyTorch TensorDataset containing the HOG features and their corresponding labels.
    """
    features_list=[]
    labels_list=[]

    for data, labels in dataloader:
        for image in data:

            image = image.numpy().reshape(28,28)
            features = hog(image)
            features_list.append(features)
        
        labels_list.append(labels)
    

    return torch.utils.data.TensorDataset(
        torch.from_numpy(np.array(features_list, dtype=np.float32)),
        torch.cat(labels_list)
    )

def compute_fourier_descriptors(image):
    """
    Compute the fourier descriptors of an image.

    Parameters:
    - image: A 2D numpy array representing the image.

    Returns:
    A 1D numpy array containing the fourier descriptors of the image.
    """

    # We first threshold the image to get the shape of the object.
    thres = threshold_otsu(image)
    _, binary = cv2.threshold(image, thres, 255, cv2.THRESH_BINARY)

    binary = binary.astype(np.uint8)
    
    # Find the contours of the object
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Find the largest contour, which is the one that encloses the object
    largest_contour = max(contours, key=cv2.contourArea)
    contour_array = largest_contour[:, 0, :]

    # Convert the contour from 2D to complex
    contour_complex = contour_array[:, 0] + 1j * contour_array[:, 1]

    # Compute the fourier descriptors
    fourier_descriptors = fft(contour_complex)

    #Take the 100 most important descriptors
    fourier_descriptors = fourier_descriptors[:100]
    
    #Apply zero padding
    if len(fourier_descriptors) < 100:
        fourier_descriptors = np.pad(fourier_descriptors, (0, 100 - len(fourier_descriptors)), 'constant')

    # The fourier descriptors are complex numbers, so we take the real and imaginary parts separately
    real_fd = np.real(fourier_descriptors)
    imag_fd = np.imag(fourier_descriptors)
    fourier_descriptors = np.concatenate((real_fd, imag_fd))

    return np.array(fourier_descriptors, dtype=np.float32)

