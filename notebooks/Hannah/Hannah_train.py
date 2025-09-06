from Hannah_pred_and_eval import get_prediction, evaluate_model
import torch

def train_model(model, train_loader, val_loader, loss_fn, optimizer, device, epochs=10, threshold=0.5):

    # auf Gerät verschieben (CPU oder GPU)
    model = model.to(device)

    # Listen zum Speichern der Metriken
    train_losses = []
    val_accuracies = []
    val_precisions = []
    val_recalls = []
    val_roc_aucs = []

    # Über Epochen iterieren
    for epoch in range(epochs):

        # In Trainingsmodus setzen
        model.train()

        # Über Mini-Batches iterieren
        running_loss = 0
        for data, target in train_loader:

            # Daten auf Gerät verschieben
            data, target = data.to(device), target.to(device)

            # Forward
            scores = model(data)

            # Backward und Optimierung
            optimizer.zero_grad() # Gradienten zurücksetzen
            loss = loss_fn(scores, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * data.size(0)

        # In Evaluierungsmodus wechseln
        model.eval()
        with torch.no_grad():  
            # Evalusierung mit Validierungsdaten nach jeder Epoche
            true_labels, pred_labels, pred_probs = get_prediction(model,val_loader, device, threshold = threshold)
            accuracy, precision, recall, roc_auc = evaluate_model(true_labels, pred_labels, pred_probs)
            print(f"Epoch {epoch + 1}: ROC = {roc_auc:.4f}, Accuracy = {accuracy:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}")   
        
        

        # Metriken speichern
        val_accuracies.append(accuracy)
        val_precisions.append(precision)
        val_recalls.append(recall)
        val_roc_aucs.append(roc_auc)

        # Duechshcnittlichen loss der Epoche berehcnen
        epoch_loss = running_loss / len(train_loader.dataset)
        # Zu Liste hinzufügen
        train_losses.append(epoch_loss)

    metrics = {
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'val_precisions': val_precisions,
        'val_recalls': val_recalls,
        'val_roc_aucs': val_roc_aucs
    }

    return metrics