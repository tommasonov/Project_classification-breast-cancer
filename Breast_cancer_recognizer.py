class NeuralNetwork:
    
    
    """
    Questa classe rappresenta una rete neurale artificiale 
    per eseguire l'addestramento usare .fit(X, y)
    per eseguire delle predizioni usare .predict(x)
    """

    def __init__(self, hidden_layer_size=100):
    
        """Costruttore della classe
        
        Args:
            hidde_layer_size(init, optional): numero di nodi dello strato nascosto
        """
        
        self.hidden_layer_size = hidden_layer_size
        
    def _init_weights(self, input_size, hidden_size):
        
        """Inizializzazione dei pesi
        
        Questa funzione inizializza i pesi a valori casuali e i bias a zero
        
        Args:
            input_size (int): numero di nodi dello strato di input
            hidden_size(int): numero di nodi dello strato nascosto
        """
    
        
        self._W1 = np.random.randn(input_size, hidden_size)
        self._b1 = np.zeros(hidden_size)
        self._W2 = np.random.randn(hidden_size, 1)
        self._b2 = np.zeros(1)
        
        
    def _accuracy(self, y_true, y_pred):
        
        """Calcolo dell'accuracy
        
        L'accuracy è una funzione di costo che ritorna la percentuale di classificazioni eseguite correttamente
        
        Args:
            y_true(ndarray): valori corretti delle osservazioni
            y_pred(ndarray): predizioni fornite dalla rete
            
        Returns:
            (float): valore dell'accuracy
        """
        
        return np.sum(y_true==y_pred)/len(y_true)
    
        
    def _log_loss(self, y_true, y_proba):
                
        """Calcolo della Log Loss (o Cross Entropy)
                
            La Log loss è una funzione di costo
            che tiene conto anche della probablilità
                
            Args:
                y_true (ndarray): valori corretti delle osservazioni
                y_proba (ndarray): output della rete in forma di probabilità
                    
            Returns:
                (float): valore della log loss
            """
        
        
        return -np.sum(np.multiply(y_true, np.log(y_proba))+
                       np.multiply((1.0001-y_true), np.log(1.0001-y_proba)))/len(y_true)
    
    
    
    def _sigmoid(self, Z):
        
        """Calcolo della Sigmoide
        La Sigmoide è una funzione di attivazione utilizzata negli strati di output per problemi
        di classificazione binaria
        
        Args:
            Z (ndarray): output lineare di uno strato
            
        Returns:
            (ndarray): output post-attivazione dello strato
        """
        
        return 1/(1+np.exp(-Z))
    
    def _relu(self, Z):
        
        """Calcolo della ReLU
        
        La ReLU è una funzione di attivazione principalmente utilizzata negli strati nascosti
        Args:
            Z (ndarray): output lineare di uno strato
            
        Returns:
            (ndarray): output post-attivazione dello strato
        
        """
        
        return np.maximum(Z,0)
    
    def _relu_derivative(self, Z):
        
        """Calcolo della derivata della ReLU
        
        La derivata della ReLU è necessaria
        per eseguire la backpropagation
        
        Args:
            Z (ndarray): output lineare di uno strato
            
        Returns:
            (ndarray): derivate parziali della ReLU rispetto a Z
        """
        
        dZ = np.zeros(Z.shape)
        dZ[Z>0] = 1
        
        return dZ
    
    def _forward_propagation(self, X):
        
        """Funzione che esegue la propagazione in avanti
        Args:
            X (ndarray): matrice con le features
            
        Returns:
            A2 (ndarray):array con l'output della rete
        """
        
        Z1 = np.dot(X, self._W1) + self._b1
        A1 = self._relu(Z1)
        Z2 = np.dot(A1, self._W2) + self._b2
        A2 = self._sigmoid(Z2)
        
        self._forward_cache = (Z1, A1, Z2, A2)
        
        return A2.flatten()
    
    def predict(self, X, return_proba=False):
        
        """Funzione che esegue la predizione
        
        Args:
            X (ndarray): matrice con le features degli esempi
            return_proba (bool,optional): se True la funzione ritorna
                anche le probabilità
                
        Returns:
            y(ndarray): array con le predizioni della rete
            proba(ndarray,optional):array con le probabilità delle predizioni
        """
        
        proba =self._forward_propagation(X)
        
        y = np.zeros(X.shape[0])
        
        y[proba>=0.5] = 1
        y[proba<0.5] = 0
        
        if(return_proba):
            return(y, proba)
        
        return y
    
    def _back_propagation(self, X, y):
        
        """Funzione che esegue la Backpropagation
        
        Args:
            X (ndarray): matrice con le features degli esempi
            y (ndarray): vettore con il target degli esempi
            
        Returns:
            dW1 (ndarray): derivate parziali della funzione di costo
                rispetto ai pesi dello strato di input
            db1 (ndarray): derivate parziali della funzione di costo 
                rispetto ai bias dello strato di input
            dW2 (ndarray): derivate parziali della funzione di costo
                rispetto ai pesi dello strato nascosto
            db2 (ndarray): derivate parziali della funzione di costo 
                rispetto ai bias dello strato nascosto
                
        """
        
        Z1, A1, Z2, A2 = self._forward_cache
        
        m = A1.shape[1]
        
        dZ2 = A2 - y.reshape(-1,1)
        dW2 = np.dot(A1.T, dZ2)/m
        db2 = np.sum(dZ2, axis=0)/m
        
        dZ1 = np.dot(dZ2, self._W2.T)*self._relu_derivative(Z1)
        dW1 = np.dot(X.T, dZ1)/m
        db1 = np.sum(dZ1, axis=0)/m
        
        return dW1, db1, dW2, db2
    
    
    def fit(self, X, y, epochs=200, lr=0.01):
        
        """Funzione che esegue il gradient Descent
         
         Args:
            X (ndarray): matrice con le features degli esempi
            y (ndarray): vettore con il target degli esempi
            epochs (int, optional): numero di epoche per il gradient descent
            lr (float, optional): valore del learning rate
        
        """
        
        self._init_weights(X.shape[1], self.hidden_layer_size)
        
        for _ in range(epochs):
            
            Y = self._forward_propagation(X)
            dW1, db1, dW2, db2 = self._back_propagation(X,y)
            
            self._W1-=lr*dW1
            self._b1-=lr*db1
            self._W2-=lr*dW2
            self._b2-=lr*db2
            
            
    def evaluate(self, X, y):
    
        """Funzione che valuta la rete calcolando le metriche
        
        Args:
            X (ndarray): matrice con le features degli esempi
            y (ndarray): vettore con il target degli esempi
            
        Returns:
            accuracy(float): valure dell'Accuracy
            log_loss(float): valore della Log Loss
        """
    
    
        y_pred, y_proba = self.predict(X, return_proba=True)
        accuracy = self._accuracy(y, y_pred)
        log_loss = self._log_loss(y, y_proba)
        
        return (accuracy,log_loss)

import pandas as pd

CSV_URL = "https://raw.githubusercontent.com/ProfAI/tutorials/master/Come%20Creare%20una%20Rete%20Neurale%20da%20Zero/breast_cancer.csv"
breast_cancer = pd.read_csv(CSV_URL)

type(breast_cancer)

breast_cancer.head()

X = breast_cancer.drop("malignant", axis=1).values
y = breast_cancer["malignant"].values

X.shape

import numpy as np

def train_test_split(X, y, test_size=0.3, random_state=None):

      if(random_state!=None):
        np.random.seed(random_state)

      n = X.shape[0]

      test_indices = np.random.choice(n, int(n*test_size), replace=False) # selezioniamo gli indici degli esempi per il test set

      # estraiamo gli esempi del test set
      # in base agli indici

      X_test = X[test_indices]
      y_test = y[test_indices]

      # creiamo il train set
      # rimuovendo gli esempi del test set
      # in base agli indici

      X_train = np.delete(X, test_indices, axis=0)
      y_train = np.delete(y, test_indices, axis=0)

      return (X_train, X_test, y_train, y_test )

X_train, X_test, y_train, y_test  = train_test_split(X, y)

X_train.shape

X_test.shape

X_max = X_train.max(axis=0)
X_min = X_train.min(axis=0)

X_train = (X_train-X_min)/(X_max-X_min)
X_test = (X_test-X_min)/(X_max-X_min)

model = NeuralNetwork()
model.fit(X_train, y_train, epochs=500)
model.evaluate(X_test, y_test)

exams_df = pd.read_csv("https://raw.githubusercontent.com/ProfAI/tutorials/master/Come%20Creare%20una%20Rete%20Neurale%20da%20Zero/exam%20results.csv")

X_new= exams_df.values
X_new = (X_new-X_min)/(X_max-X_min)

y_pred, y_proba = model.predict(X_new, return_proba=True)

classes = ["benigno", "maligno"]

for i, (pred,proba) in enumerate(zip(y_pred, y_proba)):
    print("Esame %d - Esito = %s (probabilità: %.4f %%)" % (i+1, classes[int(pred)], abs(1-pred-proba)*100))