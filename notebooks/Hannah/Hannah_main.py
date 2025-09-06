import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent 
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT)) 

from notebooks.Hannah.Hannah_NN1 import NN_1
from notebooks.Hannah.Hannah_train import train_model
from notebooks.Hannah.Hannah_pred_and_eval import get_prediction, evaluate_model
from dataloader import create_dataloaders_numeric
from notebooks.Hannah.Hannah_data_prep import data_preparation, get_subtrain_splits, FeatureEngineerer
from sklearn.datasets import fetch_openml
import torch
import torch.optim as optim 
from torch.optim import SGD  
import torch.nn as nn  
import matplotlib.pyplot as plt
from src.train_test_split import create_unified_test_split

# Daten laden 

dataset = fetch_openml(data_id=45566, as_frame=True)
st = dataset.frame

# Einheitlichen Testsplit machen
data_train_and_val, data_test = create_unified_test_split(st)

# Erneuter Split für Validierungsdaten
data_train, data_val = create_unified_test_split(data_train_and_val)     

# Daten in float32 umwandlen und target trennen und in 0 und 1 -kategorie
# x_train, y_train = data_preparation(data_train)
x_train, y_train = data_preparation(data_train)
x_val, y_val = data_preparation(data_val)
x_test, y_test = data_preparation(data_test)   

# Feature engineering
fe = FeatureEngineerer()
x_train = fe.fit_transform(x_train)
x_val = fe.transform(x_val)
x_test = fe.transform(x_test)

# Trainingsdaten in versch. große Subdataframes teilen für trianings und Modellauswahl
percentages = [0.1, 0.2, 0.3, 0.4, 0.6, 0.8]
splits = get_subtrain_splits(x_train, y_train, percentages)
X_sub_10, y_sub_10 = splits[0.1]
X_sub_20, y_sub_20 = splits[0.2]
X_sub_30, y_sub_30 = splits[0.3]
X_sub_40, y_sub_40 = splits[0.4]
X_sub_60, y_sub_60 = splits[0.6]
X_sub_80, y_sub_80 = splits[0.8]

# Seed für torch setzen
torch.manual_seed(42)


#########################################
##### ertsets trianing mit 30% der Daten
#########################################

# Daten mit Dataloader vorbereiten
trian_loader_30, val_loader_30 = create_dataloaders_numeric(X_sub_30, y_sub_30, 32 )


# Modell Parmater festlegen
feature_n = X_sub_30.shape[1]
hidden_dim1 = 300
hidden_dim2 = 200
hidden_dim3 = 70

# Device automatisch anpassen wenn GPU verfügbar 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Modell Objekt erstellen
nn1 = NN_1(feature_n, hidden_dim1, hidden_dim2, hidden_dim3)
nn1.to(device)

# parameter fürs training festlegen
epochs = 20
learning_rate = 0.001
loss_fn = nn.BCELoss() # Binary Class Entropy
optimizer = optim.Adam(nn1.parameters(), lr= learning_rate) #Adam optimizer 


# Training starten
metrics = train_model(nn1, trian_loader_30, val_loader_30,loss_fn, optimizer,device= device, epochs = epochs)


# FUnktion zum Ergebnisse visualiseren
def plot_metrics(metrics):

    epochs = range(1, len(metrics['train_losses']) + 1)

    plt.figure(figsize=(15, 10))

    # Plot des Trainingsverlusts
    plt.subplot(2, 1, 1)
    plt.plot(epochs, metrics['train_losses'], 'b-o', label='Trainingsverlust')
    plt.title('Trainingsverlauf: Verlust pro Epoche')
    plt.xlabel('Epoche')
    plt.ylabel('Verlust')
    plt.legend()
    plt.grid(True)

    # Plot der Validierungsmetriken
    plt.subplot(2, 1, 2)
    plt.plot(epochs, metrics['val_accuracies'], 'r-o', label='Validierungs-Genauigkeit')
    plt.plot(epochs, metrics['val_precisions'], 'g-o', label='Validierungs-Präzision')
    plt.plot(epochs, metrics['val_recalls'], 'c-o', label='Validierungs-Recall')
    plt.plot(epochs, metrics['val_roc_aucs'], 'm-o', label='Validierungs-ROC AUC')
    plt.title('Validierungsmetriken pro Epoche')
    plt.xlabel('Epoche')
    plt.ylabel('Wert')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Ergebnisse visualsieren
plot_metrics(metrics)


