import numpy as np
import matplotlib.pyplot as plt
import torch

class AutoEncoder:
    def __init__(self, input_dim, hidden_dim, encode_layers=1, decode_layers=1):
        self.encode_layers = encode_layers
        self.decode_layers = decode_layers
        self.encode_dims = np.linspace(input_dim, hidden_dim, encode_layers + 1,dtype=int)
        self.decode_dims = np.linspace(hidden_dim, input_dim, decode_layers + 1,dtype=int)
    
        # Initialisation des paramètres 
        scale_encode = np.sqrt(2.0 / self.encode_dims) # centrage
        scale_decode = np.sqrt(2.0 / self.decode_dims)
        
        # Parametres des autres couches 
        self.encode_parameters = [(torch.randn(self.encode_dims[i], self.encode_dims[i+1])* scale_encode[i]).clone().detach().requires_grad_(True)
                                for i in range(self.encode_layers)]
        self.encode_bias = [torch.zeros((self.encode_dims[i+1],), requires_grad=True) 
                        for i in range(self.encode_layers)]
        self.decode_parameters = [(torch.randn(self.decode_dims[i], self.decode_dims[i+1])* scale_decode[i]).clone().detach().requires_grad_(True)
                                for i in range(self.decode_layers)]
        self.decode_bias = [torch.zeros((self.decode_dims[i+1],), requires_grad=True) 
                        for i in range(self.decode_layers)]
        
        # Collecter tous les paramètres
        self.params = []
        self.params.extend(self.encode_parameters)
        self.params.extend(self.encode_bias)
        self.params.extend(self.decode_parameters)
        self.params.extend(self.decode_bias)
        
        # Initialiser les états pour chaque optimiseur (sgd classique n'en a pas besoin)
        self.init_optimizer_states()

    def init_optimizer_states(self):
        """Initialise les états pour momentum et Adam"""
        # Pour SGD + Momentum
        self.velocities = {i: torch.zeros_like(p) for i, p in enumerate(self.params)}
        
        # Pour Adam
        self.m = {i: torch.zeros_like(p) for i, p in enumerate(self.params)}  # Premier moment (momentum)
        self.v = {i: torch.zeros_like(p) for i, p in enumerate(self.params)}  # Second moment (RMSprop)
        self.t = 0  # Compteur de pas pour bias correction

    # Encode -> compression des données
    def encode(self, X, activation_function=torch.relu):
        X = X.to(torch.float32)
        for i in range(self.encode_layers):
            X = activation_function(torch.matmul(X, self.encode_parameters[i]) + self.encode_bias[i])
        return X
    
    # Decode -> Retranscription des données
    def decode(self, X, activation_function=torch.relu):
        for i in range(self.decode_layers-1):
            X = activation_function(torch.matmul(X, self.decode_parameters[i]) + self.decode_bias[i])
        X = torch.sigmoid(torch.matmul(X, self.decode_parameters[-1]) + self.decode_bias[-1])
        return X

    # Forward pass complete
    def forward(self, X, activation_function=torch.relu):
        encoded = self.encode(X, activation_function)
        decoded = self.decode(encoded, activation_function)
        return decoded

    # MSE
    def mse_loss(self, X_pred, X):
        return torch.mean(torch.square(X - X_pred))

    # Differents optimiseurs
    def sgd_step(self, learning_rate):
        """SGD Vanilla - comme ton code original"""
        with torch.no_grad():
            for p in self.params:
                if p.grad is not None:
                    p.data -= learning_rate * p.grad
                    p.grad.zero_()

    def momentum_step(self, learning_rate, momentum=0.9):
        """SGD + Momentum"""
        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is not None:
                    # velocity = momentum * velocity + gradient
                    self.velocities[i] = momentum * self.velocities[i] + p.grad
                    # param = param - lr * velocity
                    p.data -= learning_rate * self.velocities[i]
                    p.grad.zero_()

    def adam_step(self, learning_rate, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        
        Adam = Adaptive Moment Estimation
        Combine:
        - Momentum (premier moment)
        - RMSprop (second moment - adaptation du LR)
        - Bias correction pour les premiers pas
        
        Args:
            learning_rate: taux d'apprentissage (souvent 0.001 pour Adam)
            beta1: coefficient pour le momentum (typiquement 0.9)
            beta2: coefficient pour RMSprop (typiquement 0.999)
            eps: petit terme pour éviter division par zéro
        """
        self.t += 1  # Incrémenter le compteur de pas
        
        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is not None:
                    grad = p.grad
                    
                    # Étape 1: Mettre à jour le premier moment (momentum)
                    # m = beta1 * m + (1 - beta1) * gradient
                    self.m[i] = beta1 * self.m[i] + (1 - beta1) * grad
                    
                    # Étape 2: Mettre à jour le second moment (variance du gradient)
                    # v = beta2 * v + (1 - beta2) * gradient²
                    self.v[i] = beta2 * self.v[i] + (1 - beta2) * (grad ** 2)
                    
                    # Étape 3: Bias correction (important au début!)
                    # Au début, m et v sont biaisés vers 0 car initialisés à 0
                    m_hat = self.m[i] / (1 - beta1 ** self.t)
                    v_hat = self.v[i] / (1 - beta2 ** self.t)
                    
                    # Étape 4: Mise à jour du paramètre
                    # param = param - lr * m_hat / (sqrt(v_hat) + eps)
                    p.data -= learning_rate * m_hat / (torch.sqrt(v_hat) + eps)
                    
                    p.grad.zero_()

    def train(self, X, learning_rate, epochs, optimizer='adam', **kwargs):
        """
        Entraîner avec différents optimiseurs
        
        Args:
            X: données d'entraînement
            learning_rate: taux d'apprentissage
            epochs: nombre d'époques
            optimizer: 'sgd', 'momentum', ou 'adam'
            **kwargs: arguments supplémentaires pour l'optimiseur
        """
        
        n = X.shape[0]
        chunk_nb = n // kwargs.get('batch_size' , 200)
        chunks = torch.chunk(X, chunk_nb)

        for epoch in range(epochs):
            # Forward pass
            indice = np.random.randint(chunk_nb)

            X_batched = chunks[indice]

            X_pred_batched = self.forward(X_batched)
            loss = self.mse_loss(X_pred_batched, X_batched)
            
            # Backward pass
            loss.backward()
            
            # Optimizer step
            if optimizer == 'sgd':
                self.sgd_step(learning_rate)
            elif optimizer == 'momentum':
                momentum = kwargs.get('momentum', 0.9)
                self.momentum_step(learning_rate, momentum)
            elif optimizer == 'adam':
                beta1 = kwargs.get('beta1', 0.9)
                beta2 = kwargs.get('beta2', 0.999)
                eps = kwargs.get('eps', 1e-8)
                self.adam_step(learning_rate, beta1, beta2, eps)
            else:
                raise ValueError(f"Optimiseur inconnu: {optimizer}")
            
            if epoch % 100 == 0:
                print(f"[{optimizer.upper()}] Epoch {epoch:4d}, Loss: {loss.item():.6f}")


