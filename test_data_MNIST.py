from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from auto_encoder_class import AutoEncoder

# Utiliser un transform pour normaliser les données en tensor
transform = transforms.Compose([
    transforms.ToTensor()
])

# Le jeu de données MNIST est composé d'images 28x28 pixels en niveaux de gris.
mnist_train = datasets.MNIST(
    root='./data',  # Chemin vers la racine des données
    train=True,      # Utiliser les données d'entraînement (pour mnist_train)
    download=True,
    transform=transform    
)

mnist_test = datasets.MNIST(
    root='./data',  # Chemin vers la racine des données
    train=False,      # Utiliser les données de test (pour mnist_test)
    download=True,
    transform=transform     
)

# Visualiser quelques images du jeu de données MNIST
fig, axs = plt.subplots(2, 5, figsize=(10, 4))
for i in range(10):
    idx = np.random.randint(len(mnist_train))
    img, label = mnist_train[idx]

    axs[i // 5, i % 5].imshow(img.squeeze(), cmap='gray')  # Afficher l'image en niveaux de gris
    axs[i // 5, i % 5].set_title(f'Label: {label}') 
    axs[i // 5, i % 5].axis('off')  # Désactiver les axes

plt.savefig("mnist_images.png")  

ae = AutoEncoder(input_dim=784, hidden_dim=36, encode_layers=1, decode_layers=1)  # Créer une instance de l'autoencodeur

X_train = mnist_train.data
X_train = X_train.reshape(-1, 784) / 255  # Reshape les données d'entraînement pour qu'elles soient compatibles avec l'autoencodeur
X_test = mnist_test.data
X_test = X_test.reshape(-1, 784) / 255

ae.train(X_train,0.005,5000)  # Entraîner l'autoencodeur sur les données d'entraînement

# Test l'autoencodeur sur des données de test
m = int(5)
encoded_test = ae.encode(X_test[:m])
decoded_test = ae.decode(encoded_test)

fig, axs = plt.subplots(3, 5, figsize=(10, 4))
for i in range(m):
    axs[0, i].imshow(X_test[i].reshape(28, 28), cmap='gray')
    axs[1, i].imshow(encoded_test[i].detach().numpy().reshape(6,6), cmap='gray')
    axs[2, i].imshow(decoded_test[i].detach().numpy().reshape(28, 28), cmap='gray')
    axs[0, i].axis('off')
    axs[1, i].axis('off')    
    axs[2, i].axis('off')

plt.savefig("autoencoder_test_result.png")  # Enregistrer la figure sous le nom "autoencoder_result.png"
