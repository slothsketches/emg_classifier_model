import torch
import torch.nn as nn


class EMGClassifier(nn.Module):
    def __init__(self, seq_len: int = 150):
        super().__init__()
        self.seq_len = seq_len

        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.fc = nn.Linear(16 * (self.seq_len // 2), 3)

    def forward(self, x: torch.Tensor):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)


if __name__ == "__main__":
    import click

    @click.group()
    def __cli__():
        pass

    @__cli__.command("train")
    def __cli_train__():
        import pathlib

        from torch.utils.data import DataLoader

        from .dataset import EMGDataset

        dataset = EMGDataset(pathlib.Path("./emg_data/"))
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = EMGClassifier().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(100):
            for x_batch, y_batch in dataloader:
                x_batch: torch.Tensor
                y_batch: torch.Tensor

                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                logits = model(x_batch)
                loss: torch.Tensor = loss_fn(logits, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch} loss: {loss.item():.4f}")

        torch.save(model.state_dict(), "./emg_data_model.torch_model")

    @__cli__.command("predict")
    def __cli_predict__():
        import pathlib

        import numpy as np
        import torch.nn.functional as F

        from .utils import load_from_sample

        file = pathlib.Path("./emg_data/measurement_open_hand.txt")
        device = "cuda:0"

        model = EMGClassifier()
        model.load_state_dict(
            torch.load("./emg_data_model.torch_model", weights_only=True)
        )
        model = model.to("cuda:0")

        model.eval()

        samples = np.array(list(load_from_sample(file)))

        sample_size = len(samples)

        if sample_size < model.seq_len:
            samples = np.pad(samples, (0, model.seq_len - sample_size), mode="constant")

        offset = -(sample_size % model.seq_len)
        splitted_samples = map(
            lambda data: torch.tensor(data, dtype=torch.float32),
            np.split(samples[:offset], sample_size // model.seq_len),
        )

        inputs = torch.stack(list(splitted_samples)).unsqueeze(1).to(device)

        with torch.no_grad():
            outputs = model(inputs)
            probabilities = F.softmax(outputs, dim=1)
            avg_probs = torch.mean(probabilities, dim=0).cpu().numpy()

        print([float(_) for _ in avg_probs])

    __cli__()
