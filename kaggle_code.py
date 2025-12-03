import json
from glob import glob
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from jiwer import wer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

CONFIG = {
    "data_dir": "/kaggle/input/brain-to-text-25/t15_copyTask_neuralData/hdf5_data_final/",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "batch_size": 32,
    "num_epochs": 50,
    "hidden_dim": 512,
    "num_layers": 3,
    "dropout": 0.3,
    "learning_rate": 1e-3,
}
print(f"Device: {CONFIG['device']}")
print(f"PyTorch version: {torch.__version__}")


def load_split(data_dir, split="train"):
    pattern = f"{data_dir}/**/data_{split}.hdf5"
    files = sorted(glob(pattern, recursive=True))
    print(f"\nLoading {split} split...")
    print(f"Files found: {len(files)}")
    all_data = {
        k: []
        for k in [
            "neural",
            "n_steps",
            "sentence",
            "phonemes",
            "phoneme_len",
            "session",
            "block",
            "trial",
        ]
    }
    for filepath in tqdm(files):
        with h5py.File(filepath, "r") as f:
            for trial_key in f.keys():
                trial = f[trial_key]
                neural = trial["input_features"][:]
                n_steps = trial.attrs["n_time_steps"]
                session = trial.attrs["session"]
                if isinstance(session, bytes):
                    session = session.decode("utf-8")
                block = trial.attrs["block_num"]
                trial_num = trial.attrs["trial_num"]
                sentence = trial.attrs.get("sentence_label")
                if sentence and isinstance(sentence, bytes):
                    sentence = sentence.decode("utf-8")
                phonemes = (
                    trial.get("seq_class_ids")[:] if "seq_class_ids" in trial else None
                )
                phoneme_len = trial.attrs.get("seq_len")
                all_data["neural"].append(neural)
                all_data["n_steps"].append(n_steps)
                all_data["sentence"].append(sentence)
                all_data["phonemes"].append(phonemes)
                all_data["phoneme_len"].append(phoneme_len)
                all_data["session"].append(session)
                all_data["block"].append(block)
                all_data["trial"].append(trial_num)
    print(f"✓ Loaded {len(all_data['neural'])} samples")
    return all_data


class BrainToTextDataset(Dataset):
    def __init__(self, data, char2idx=None, normalize=True):
        self.neural = data["neural"]
        self.n_steps = data["n_steps"]
        self.sentences = data["sentence"]
        self.normalize = normalize
        if char2idx is None:
            self.char2idx = self._build_vocab()
        else:
            self.char2idx = char2idx
        self.idx2char = {v: k for k, v in self.char2idx.items()}
        self.vocab_size = len(self.char2idx)

    def _build_vocab(self):
        chars = set()
        for sent in self.sentences:
            if sent:
                chars.update(sent.lower())
        chars = sorted(list(chars))
        char2idx = {"<BLANK>": 0}
        for i, ch in enumerate(chars, start=1):
            char2idx[ch] = i
        return char2idx

    def __len__(self):
        return len(self.neural)

    def __getitem__(self, idx):
        neural = self.neural[idx][: self.n_steps[idx]]
        if self.normalize:
            neural = (neural - neural.mean(axis=0)) / (neural.std(axis=0) + 1e-8)
        sentence = self.sentences[idx] if self.sentences[idx] else ""
        target = [self.char2idx.get(ch.lower(), 0) for ch in sentence]
        return {
            "neural": torch.FloatTensor(neural),
            "target": torch.LongTensor(target),
            "length": len(neural),
            "target_length": len(target),
            "sentence": sentence,
        }


def collate_fn(batch):
    batch = sorted(batch, key=lambda x: x["length"], reverse=True)
    neurals = [item["neural"] for item in batch]
    targets = [item["target"] for item in batch]
    neural_padded = pad_sequence(neurals, batch_first=True)
    target_padded = pad_sequence(targets, batch_first=True)
    lengths = torch.LongTensor([item["length"] for item in batch])
    target_lengths = torch.LongTensor([item["target_length"] for item in batch])
    return {
        "neural": neural_padded,
        "target": target_padded,
        "lengths": lengths,
        "target_lengths": target_lengths,
        "sentences": [item["sentence"] for item in batch],
    }


class BaselineCTCModel(nn.Module):
    def __init__(
        self, input_dim=512, hidden_dim=512, num_layers=3, vocab_size=50, dropout=0.3
    ):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            256,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x, lengths):
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        x_packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        lstm_out, _ = self.lstm(x_packed)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        logits = self.fc(lstm_out)
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs.transpose(0, 1)


def validate_model(model, val_loader, idx2char, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            neural = batch["neural"].to(device)
            lengths = batch["lengths"]
            log_probs = model(neural, lengths)
            _, max_indices = log_probs.max(dim=-1)
            for b in range(max_indices.size(1)):
                seq = max_indices[: lengths[b], b].cpu().numpy()
                decoded = []
                prev = None
                for token in seq:
                    if token != 0 and token != prev:
                        decoded.append(idx2char.get(token, ""))
                    prev = token
                all_preds.append("".join(decoded))
            all_targets.extend(batch["sentences"])
    try:
        error_rate = wer(
            [t.lower().strip() for t in all_targets],
            [p.lower().strip() for p in all_preds],
        )
    except:
        error_rate = 1.0
    return error_rate * 100


def train_model(train_loader, val_loader, char2idx, config):
    model = BaselineCTCModel(
        input_dim=512,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        vocab_size=len(char2idx),
        dropout=config["dropout"],
    ).to(config["device"])
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["learning_rate"],
        epochs=config["num_epochs"],
        steps_per_epoch=len(train_loader),
    )
    idx2char = {v: k for k, v in char2idx.items()}
    best_wer = float("inf")
    print("\nTraining...")
    for epoch in range(config["num_epochs"]):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["num_epochs"]}'):
            neural = batch["neural"].to(config["device"])
            target = batch["target"].to(config["device"])
            lengths = batch["lengths"]
            target_lengths = batch["target_lengths"]
            log_probs = model(neural, lengths)
            loss = criterion(log_probs, target, lengths, target_lengths)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        avg_loss = train_loss / len(train_loader)
        val_wer = validate_model(model, val_loader, idx2char, config["device"])
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, WER={val_wer:.2f}%")
        if val_wer < best_wer:
            best_wer = val_wer
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "char2idx": char2idx,
                    "config": config,
                    "wer": val_wer,
                },
                "best_model.pt",
            )
            print(f"✓ Saved best model (WER: {val_wer:.2f}%)")
    return model, best_wer


def load_test_data_for_submission(data_dir):
    print("\nLoading test data for submission...")
    pattern = f"{data_dir}/**/data_test.hdf5"
    files = sorted(glob(pattern, recursive=True))
    all_samples = []
    sample_id = 0
    for filepath in tqdm(files):
        session = Path(filepath).parent.name
        with h5py.File(filepath, "r") as f:
            trial_keys = sorted([k for k in f.keys() if "trial" in k.lower()])
            for trial_key in trial_keys:
                trial = f[trial_key]
                if "input_features" not in trial:
                    continue
                features = trial["input_features"][:]
                n_steps = trial.attrs["n_time_steps"]
                features = features[:n_steps]
                features = (features - features.mean(axis=0)) / (
                    features.std(axis=0) + 1e-8
                )
                features = np.clip(features, -5, 5)
                all_samples.append(
                    {
                        "id": sample_id,
                        "session": session,
                        "trial_key": trial_key,
                        "features": torch.FloatTensor(features),
                        "length": len(features),
                    }
                )
                sample_id += 1
    print(f"✓ Loaded {len(all_samples)} test samples")
    all_samples.sort(key=lambda s: (s["session"], s["trial_key"]))
    return all_samples


def generate_predictions(model, test_samples, idx2char, device, batch_size=32):
    model.eval()
    indexed_predictions = {}
    print("\nGenerating predictions...")
    batches = [
        test_samples[i : i + batch_size]
        for i in range(0, len(test_samples), batch_size)
    ]
    with torch.no_grad():
        for batch in tqdm(batches):
            original_indices = [s["id"] for s in batch]
            features = [s["features"] for s in batch]
            lengths = torch.LongTensor([s["length"] for s in batch])
            features_padded = pad_sequence(features, batch_first=True).to(device)
            sorted_lengths, sorted_idx = lengths.sort(descending=True)
            features_sorted = features_padded[sorted_idx]
            log_probs = model(features_sorted, sorted_lengths)
            _, max_indices = log_probs.max(dim=-1)
            batch_preds = []
            for b in range(max_indices.size(1)):
                seq = max_indices[: sorted_lengths[b], b].cpu().numpy()
                decoded = []
                prev = None
                for token in seq:
                    if token != 0 and token != prev:
                        decoded.append(idx2char.get(token, ""))
                    prev = token
                batch_preds.append("".join(decoded))
            unsorted_preds = [""] * len(batch_preds)
            for i, pred in enumerate(batch_preds):
                original_batch_index = sorted_idx[i].item()
                unsorted_preds[original_batch_index] = pred
            for original_id, prediction in zip(original_indices, unsorted_preds):
                indexed_predictions[original_id] = prediction
    predictions = [indexed_predictions[i] for i in range(len(test_samples))]
    return predictions


def create_submission_file(predictions, output_path="submission.csv"):
    print("\nCreating submission file...")
    df = pd.DataFrame({"id": range(len(predictions)), "text": predictions})
    df.to_csv(output_path, index=False)
    print(f"\n✓ Submission saved to: {output_path}")
    print(f"  Total predictions: {len(df)}")
    print("\nSample predictions:")
    print(df.head(10).to_string(index=False))
    return df


def main():
    print("\n[1/5] Loading Train/Validation data...")
    train_data = load_split(CONFIG["data_dir"], "train")
    val_data = load_split(CONFIG["data_dir"], "val")
    print("\n[2/5] Creating Datasets and DataLoaders...")
    train_dataset = BrainToTextDataset(train_data)
    val_dataset = BrainToTextDataset(val_data, char2idx=train_dataset.char2idx)
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    print(f"Vocab size: {train_dataset.vocab_size}")
    print(
        f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}"
    )
    print("\n[3/5] Training model...")
    model, best_wer = train_model(
        train_loader, val_loader, train_dataset.char2idx, CONFIG
    )
    print(f"\n✓ Training complete! Best Validation WER: {best_wer:.2f}%")
    print("\n[4/5] Loading Test data for submission...")
    test_samples = load_test_data_for_submission(CONFIG["data_dir"])
    print("\n[5/5] Creating submission...")
    print("Loading best_model.pt...")
    checkpoint = torch.load("best_model.pt", map_location=CONFIG["device"])
    model_config = checkpoint.get("config", CONFIG)
    model = BaselineCTCModel(
        input_dim=512,
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        vocab_size=len(checkpoint["char2idx"]),
        dropout=model_config["dropout"],
    ).to(CONFIG["device"])
    model.load_state_dict(checkpoint["model_state_dict"])
    idx2char = {v: k for k, v in checkpoint["char2idx"].items()}
    predictions = generate_predictions(
        model, test_samples, idx2char, CONFIG["device"], batch_size=32
    )
    submission = create_submission_file(predictions, "submission.csv")


if __name__ == "__main__":
    Path("checkpoints").mkdir(exist_ok=True)
    main()
