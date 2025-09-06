import torch
import torch.nn as nn               
import torch.nn.functional as F     


class NN_1(nn.Module):

    def __init__(self, feature_n, hidden_dim1, hidden_dim2, hidden_dim3):

        #Konstruktor der Elternklasse aufrufen
        super(NN_1, self).__init__()

        # Normalisierungsschicht 
        self.bn = nn.BatchNorm1d(feature_n)

        # fully connected layer 
        # soll immer ein orginal feature mit dem dazugehrigen Feature verarebiten das angibt ob der Wert einzigartig ist
        self.fc1 = nn.Linear(2, hidden_dim1) # Parallele Verarbeitung

         # Zusätzliche hidden-layers  für die globale Verarbeitung
        self.fc2 = nn.Linear(200 * hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)

        # Ausgabe layer
        self.ol = nn.Linear(hidden_dim3, 1) 

        self.sigmoid = nn.Sigmoid()






    def forward(self, x):

        # Sample size 
        N = x.shape[0]

        # Batch Normalisierung
        x = self.bn(x)
 

        # Aufteilen der Originalfeatures und der engineerten Features
        # Dimensionen: [batch_size, 200, 1] 
        x_orig = x[: , :200]
        x_engineered = x[:, 200:]

        # Dimensionen anpassen für das zusammenfügen
        x_orig = x_orig.unsqueeze(2)
        x_engineered = x_engineered.unsqueeze(2)

        # jetzt zusammenfügen in [batch_size, 200, 2]
        x = torch.cat([x_orig, x_engineered], dim=2)

        # Parallele Verarbeitung der Feature Paare in erster fully connected layer 
        x = F.relu(self.fc1(x))

        # reshape  um erwartete 2 Dimensionen zu erhalten
        x = x.view(N, -1)

        # Zusätzliche fully connected layer für die globale Mustererkennung
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # Ausgabe layer
        x = self.ol(x)

        #  Sigmoid Aktivierungsfunktion und reshape zu [batch_size]
        x = self.sigmoid(x)

        return x
    
    

