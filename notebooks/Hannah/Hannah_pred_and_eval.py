import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

def get_prediction(model, loader, device, threshold=0.5):

    # Evaluierungsmodus
    model.eval() 

    # Klassenvorhersage
    pred_labels =[]

    # Tatsächliche Labels
    true_labels = []

    # geschätzte Wahrscheinlichkeiten für prognostizierte Klasse
    pred_probas = []

    # Deaktiviert die Gradientenberechnung weil das nur bei Backpropagation gebraucht wird
    with torch.no_grad(): 
        for X_batch, y_batch in loader:

            # Daten ggf auf GPU verschieben
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Vorhersage
            scores = model(X_batch)

            # Vorhersage in Liste speichern
            pred_probas += scores.tolist()

            # Wahre Labels in Liste speichern
            true_labels += y_batch.tolist()

            # Binäre Klassenvorhersage basierend auf dem Schwellenwert
            preds = (scores >= threshold).long()

            # Vorhersagen in Liste speichern
            pred_labels += preds.tolist()

        # zurück in Trainingmodus wechseln
        model.train()

        return np.array(true_labels), np.array(pred_labels), np.array(pred_probas)




def evaluate_model( true_labels, pred_labels, pred_probs):
        
        # Metriken berechnen
        accuracy = accuracy_score(true_labels, pred_labels)
        precision = precision_score(true_labels, pred_labels, zero_division=0)
        recall = recall_score(true_labels, pred_labels, zero_division=0)
        roc_auc = roc_auc_score(true_labels, pred_probs)

        return accuracy, precision, recall, roc_auc

