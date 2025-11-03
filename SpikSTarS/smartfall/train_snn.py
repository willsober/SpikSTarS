import torch
import torch.nn as nn
import snntorch as snn
from snntorch import functional as SF
from dataloader import SmartFallDataset
from torch.utils.data import DataLoader
from utils import *
import torch.optim.lr_scheduler as lr_scheduler
from criterion import *
from model import SNNModel0HLayers
import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, \
    ConfusionMatrixDisplay

torch.manual_seed(111)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "mps") if torch.backends.mps.is_available() else torch.device("cpu")

num_steps = 25
batch_size = 4
train_dataset = SmartFallDataset('/tmp/pycharm_project_582/dataset/smart_fall/time_window_1s_sliding_100ms/train')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataset = SmartFallDataset('/tmp/pycharm_project_582/dataset/smart_fall/time_window_1s_sliding_100ms/test')
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

snn_model = SNNModel0HLayers(time_steps=num_steps, input_features=6)
snn_model = snn_model.to(device)
num_epochs = 40
optimizer = torch.optim.Adam(params=snn_model.parameters(), lr=0.01)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.05, patience=3)
loss_function = SF.mse_count_loss(correct_rate=0.7, incorrect_rate=0.3)
# loss_function = nn.CrossEntropyLoss()

# Model parameter stats
total_num = sum(p.numel() for p in snn_model.parameters())
trainable_num = sum(p.numel() for p in snn_model.parameters() if p.requires_grad)
print(f'Model statistic >>> Total params: {total_num:,} | Trainable params: {trainable_num:,}')

checkpoint_path = './saves/snn_model_1s_40ep_mse_count_loss_lc_sampling25_seed111.pt'

# For confusion matrix and metrics at the last epoch
all_preds = []
all_targets = []


def get_predictions(spk):
    """
    Get predictions based on the spike counts.
    spk: Tensor of shape [T, B, C]
    Returns: Predicted classes (Tensor of shape [B])
    """
    # Permute to get [B, C, T]
    if spk.shape[0] == num_steps:
        spk = spk.permute(1, 2, 0)
    elif spk.shape[2] != num_steps:
        raise ValueError(f"Unexpected spk shape: {spk.shape}")

    # Sum across the time dimension (T) to get the spike count for each class
    spk_sum = spk.sum(dim=2)

    # Get the class with the highest spike count
    _, pred = spk_sum.max(dim=1)  # [B] - predicted class for each sample in the batch
    return pred


with tqdm.trange(num_epochs) as pbar:
    for epoch in pbar:
        snn_model.train()
        train_acc = 0
        train_loss = 0
        val_acc = 0
        val_loss = 0

        for _, (inputs, label) in enumerate(train_loader):
            # LC Sampling
            spikes_up_input, spikes_down_input = lc_sampling(inputs)
            inputs = torch.concat((spikes_up_input, spikes_down_input), dim=1)

            inputs = inputs.to(device)
            label = label.to(device)  # Target labels
            target = torch.argmax(label, dim=1)

            spk = snn_model(inputs)  # forward-pass

            loss_val = loss_function(spk, target)
            train_loss += loss_val.item()

            optimizer.zero_grad()  # Zero out gradients
            loss_val.backward()  # Calculate gradients
            optimizer.step()  # Update weights

            acc_val = SF.accuracy_rate(spk, target)
            train_acc += acc_val

        avg_loss = train_loss / len(train_loader)
        avg_acc = train_acc / len(train_loader)

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Avg Loss: {avg_loss:.5f}, Avg Accuracy: {avg_acc:.5f}")

        snn_model.eval()
        with torch.no_grad():
            for _, (inputs, label) in enumerate(test_loader):
                # LC Sampling
                spikes_up_input, spikes_down_input = lc_sampling(inputs)
                inputs = torch.concat((spikes_up_input, spikes_down_input), dim=1)

                inputs = inputs.to(device)
                label = label.to(device)
                target = torch.argmax(label, dim=1)

                spk = snn_model(inputs)

                loss_val = loss_function(spk, target)
                val_loss += loss_val.item()

                acc_val = SF.accuracy_rate(spk, target)
                val_acc += acc_val

                # Collect predictions and targets for confusion matrix and metrics
                pred = get_predictions(spk)  # Use the custom get_predictions function
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        avg_val_loss = val_loss / len(test_loader)
        avg_val_acc = val_acc / len(test_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Avg_val_loss: {avg_val_loss:.5f}, Avg_val_acc: {avg_val_acc:.5f}")
        scheduler.step(avg_val_loss)

        if (epoch + 1) % 5 == 0:
            torch.save(snn_model.state_dict(), checkpoint_path)

        # At the last epoch, calculate the confusion matrix and metrics
        if (epoch + 1) == num_epochs:
            print("Evaluating final metrics...")

            # Confusion Matrix
            cm = confusion_matrix(all_targets, all_preds)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fall", "Not Fall"])
            disp.plot(cmap=plt.cm.Blues)
            plt.title(f"Confusion Matrix ")
            plt.show()

            # Additional metrics
            accuracy = accuracy_score(all_targets, all_preds)
            precision = precision_score(all_targets, all_preds, average='binary')
            recall = recall_score(all_targets, all_preds, average='binary')
            f1 = f1_score(all_targets, all_preds, average='binary')

            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
