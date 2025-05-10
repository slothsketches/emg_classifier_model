import pathlib
import typing as t

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split

from .model import EMGClassifierCNN
from .utils import load_from_sample_windowed, normalize_windows


class EMGDataset(Dataset):
    def __init__(
        self,
        windows: torch.Tensor,
        labels: torch.Tensor,
    ):
        self.windows = windows
        self.labels = labels

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index: int):
        return self.windows[index], self.labels[index]

    @classmethod
    def from_data_store(
        cls,
        datastore: t.Dict[int, t.Iterable[pathlib.Path]],
        window_size: int,
        stride: int,
    ):
        labels = []
        windows = []

        n = 0

        for label, paths in datastore.items():
            k = n
            for path in paths:
                for window in load_from_sample_windowed(path, window_size, stride):
                    k += 1
                    windows.append(window)
                    labels.append(label)

        return cls(
            normalize_windows(np.array(windows, dtype=np.float32)).unsqueeze(1),
            torch.tensor(labels, dtype=torch.int64),
        )


if __name__ == "__main__":
    from sklearn.metrics import classification_report, confusion_matrix

    # fmt: off
    datastore = {
        0: [
            pathlib.Path("./emg_data/measurement_closed_fist.txt"),
            # pathlib.Path("./emg_data_test/measurement_closed_fist_1.txt")
        ],
        1: [
            pathlib.Path("./emg_data/measurement_open_hand.txt"),
            # pathlib.Path("./emg_data_test/measurement_open_hand_1.txt")
        ],
        2: [
            pathlib.Path("./emg_data/measurement_pinch_hand.txt"),
            # pathlib.Path("./emg_data_test/measurement_pinch_hand_1.txt")
        ],
    }
    keys_map = {
        0: "Closed Fist",
        1: "Open Hand",
        2: "Pinch Hand",
    }
    # fmt: on

    validity_split = 0.1
    test_split = 0.3
    batch_size = 32
    window_size = 150
    stride_size = 50
    n_epochs = 100
    lr = 0.001

    dataset = EMGDataset.from_data_store(datastore, window_size, stride_size)
    dataset_size = len(dataset)

    validity_size = int(validity_split * dataset_size)
    test_size = int(test_split * dataset_size)

    print(
        f"Loaded: {dataset_size}, re-validating with {validity_split * 100:.02f}% ({validity_size}) and testing with {test_split * 100:.02f}% ({test_size})"
    )

    train_size = dataset_size - validity_size - test_size

    if not all((validity_size, test_size, train_size)):
        raise RuntimeError(
            f"Invalid sizes (validity, test, train): [{validity_size}, {test_size}, {train_size}]"
        )

    train_data, validity_data, test_data = random_split(
        dataset, [train_size, validity_size, test_size]
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validity_loader = DataLoader(validity_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    model = EMGClassifierCNN(len(datastore), window_size)
    model = model.to("cuda:0")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to("cuda:0")
            labels = labels.to("cuda:0")
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        model.eval()
        correct = 0
        total = 0
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in validity_loader:
                inputs = inputs.to("cuda:0")
                labels = labels.to("cuda:0")

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = running_val_loss / len(validity_loader)
        accuracy = 100 * correct / total
        print(
            f"Epoch [{epoch + 1}/{n_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {accuracy:.2f}%"
        )

    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_true = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to("cuda:0")
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            predicted_cpu = predicted.cpu()
            all_preds.extend(predicted_cpu.numpy())
            all_true.extend(labels.numpy())
            total += labels.size(0)
            correct += (predicted_cpu == labels).sum().item()

    test_accuracy = 100 * correct / total
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    print(
        classification_report(all_true, all_preds, target_names=list(keys_map.values()))
    )
    print(confusion_matrix(all_true, all_preds))
