import torch
from torch.optim.lr_scheduler import CyclicLR, StepLR
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from evaluate import plot_confusion_matrix, plot_metrics, plot_learning_rate

def train_and_validate(model, model_name, train_loader, validation_loader, optimizer, scheduler, criteria, num_epochs):

    print(f"Training {model_name} model")
    
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    learning_rates = []
    #l1_lambda = 10e-8

    for epoch in tqdm(range(num_epochs)):
        model.train()
        total_train_loss, total_train_correct, total_train_samples = 0, 0, 0
        
        for value in tqdm(train_loader):
            optimizer.zero_grad()
            if len(value) == 3:
                photos, flows, labels = value
                outputs = model(photos, flows)
            else:
                inputs, labels = value
                outputs = model(inputs)
            # Save in a text file all the labels
            loss = criteria(outputs, labels)
            
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)

            total_train_correct += (predicted == labels).sum().item()
            total_train_samples += labels.size(0)

            # Update learning rate after each batch for CyclicLR
            if isinstance(scheduler, CyclicLR):
                learning_rates.append(optimizer.param_groups[0]['lr'])
                scheduler.step()
        
        # Update learning rate after each epoch for Dynamic LR
        if isinstance(scheduler, StepLR):
            learning_rates.append(optimizer.param_groups[0]['lr'])
            scheduler.step()
        
        avg_train_loss = total_train_loss / total_train_samples
        train_accuracy = total_train_correct / total_train_samples
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        
        # Validation phase
        model.eval()
        total_val_loss, total_val_correct, total_val_samples = 0, 0, 0
        all_preds, all_true = [], []
        with torch.no_grad():
            for value in tqdm(validation_loader):
                if len(value) == 3:
                    photos, flows, labels = value
                    outputs = model(photos, flows)
                else:
                    inputs, labels = value
                    outputs = model(inputs)
                val_loss = criteria(outputs, labels)
                
                total_val_loss += val_loss.item() * labels.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val_correct += (predicted == labels).sum().item()
                total_val_samples += labels.size(0)

                # Collect all predictions and true labels
                all_preds.extend(predicted.view(-1).cpu().numpy())
                all_true.extend(labels.view(-1).cpu().numpy())

        avg_val_loss = total_val_loss / total_val_samples
        val_accuracy = total_val_correct / total_val_samples
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
                
        print(f'Epoch {epoch+1}/{num_epochs}, '
                f'Train Loss: {avg_train_loss:.4f}, '
                f'Val Loss: {avg_val_loss:.4f}, '
                f'Train Acc: {train_accuracy:.4f}, '
                f'Val Acc: {val_accuracy:.4f}')
    
        # Compute confusion matrix
        classes = ["clap", "climb", "drink", "jump", "pour", "ride_bike", "ride_horse", 
                "run", "shoot_bow", "smoke", "throw", "wave"]
        if isinstance(scheduler, CyclicLR):
            scheduler_type = 'cyclic'
        elif isinstance(scheduler, StepLR):
            scheduler_type = 'dynamic'
        else:
            scheduler_type = 'fixed'
        # Save the model
        torch.save(model.state_dict(), f'{model_name}_epoch_{epoch}_{scheduler_type}.pth')
        cm = confusion_matrix(all_true, all_preds)
        plot_confusion_matrix(model_name, cm, classes, scheduler_type, title=f'{epoch}', cmap=plt.cm.Blues)
        # Plot the training and validation losses and accuracies
        plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, model_name, 'Fixed LR')

    # Plot the learning rate
    plot_learning_rate(learning_rates, scheduler_type, model_name)

    # Save the model with the scheduler type
    torch.save(model.state_dict(), f'{model_name}_{scheduler_type}.pth')

    # save txt file with the model's train and validation losses and accuracies for each epoch
    with open(f'plots/{model_name}/{model_name}_losses_accuracies.txt', 'a') as f:
        for epoch in range(num_epochs):
            f.write(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[epoch]:.4f}, Val Loss: {val_losses[epoch]:.4f}, Train Acc: {train_accuracies[epoch]:.4f}, Val Acc: {val_accuracies[epoch]:.4f}\n')

    return train_losses, val_losses, train_accuracies, val_accuracies

