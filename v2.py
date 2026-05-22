import os
import ast
import random
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import librosa
import timm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

                                                              
           
                                                              

class CFG:
    DATA_DIR = Path("./birdclef-2026")
    OUTPUT_DIR = Path("./outputs_single_timm_v2")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    RUN_TRAIN = True
    RUN_INFER = True

    train_folds = [0, 1, 2]
    seed = 42

    sample_rate = 32000
    duration = 5
    audio_len = sample_rate * duration

    n_fft = 2048
    hop_length = 512
    n_mels = 128
    fmin = 20
    fmax = 16000
    power = 2.0

    model_name = "tf_efficientnetv2_b0.in1k"
    pretrained = True
    in_chans = 1
    dropout = 0.25

    n_folds = 5
    epochs = 12
    batch_size = 32
    valid_batch_size = 64
    num_workers = 2

    lr = 2e-4
    min_lr = 1e-6
    weight_decay = 1e-4
    grad_clip = 5.0
    use_amp = True

    use_pos_weight = True
    pos_weight_clip = 20.0

    use_mixup = True
    mixup_alpha = 0.4
    mixup_prob = 0.5

    use_secondary_labels = True
    use_train_soundscapes = True
    train_soundscape_weight = 5

    infer_batch_size = 64
    tta_shifts = [-1.0, 0.0, 1.0]

    debug = False
    debug_train_rows = 1200


def setup_paths():
    if not CFG.DATA_DIR.exists():
        candidates = list(Path("/kaggle/input").rglob("sample_submission.csv"))
        if candidates:
            CFG.DATA_DIR = candidates[0].parent

    CFG.TRAIN_CSV = CFG.DATA_DIR / "train.csv"
    CFG.TAXONOMY_CSV = CFG.DATA_DIR / "taxonomy.csv"
    CFG.SAMPLE_SUBMISSION = CFG.DATA_DIR / "sample_submission.csv"
    CFG.TRAIN_AUDIO_DIR = CFG.DATA_DIR / "train_audio"
    CFG.TRAIN_SOUNDSCAPES_DIR = CFG.DATA_DIR / "train_soundscapes"
    CFG.TRAIN_SOUNDSCAPES_LABELS = CFG.DATA_DIR / "train_soundscapes_labels.csv"
    CFG.TEST_SOUNDSCAPES_DIR = CFG.DATA_DIR / "test_soundscapes"


setup_paths()


                                                              
                    
                                                              

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


seed_everything(CFG.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


                                                              
            
                                                              

def parse_secondary_labels(x) -> List[str]:
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return [str(v) for v in x]
    if isinstance(x, str):
        x = x.strip()
        if x in ["", "[]"]:
            return []
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except Exception:
            pass
        if ";" in x:
            return [v.strip() for v in x.split(";") if v.strip()]
        if "," in x:
            return [v.strip() for v in x.split(",") if v.strip()]
    return []


def parse_time_to_seconds(x) -> float:
    if pd.isna(x):
        return 0.0
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    value = str(x).strip()
    if not value:
        return 0.0
    if ":" not in value:
        return float(value)
    parts = [float(part) for part in value.split(":")]
    if len(parts) == 3:
        h, m, s = parts
        return h * 3600 + m * 60 + s
    if len(parts) == 2:
        m, s = parts
        return m * 60 + s
    raise ValueError(f"Unsupported timestamp format: {x!r}")


def find_audio_path(audio_dir: Path, filename: str, label: Optional[str] = None) -> Path:
    direct = audio_dir / filename
    if direct.exists():
        return direct
    if label is not None:
        nested = audio_dir / label / filename
        if nested.exists():
            return nested
    matches = list(audio_dir.rglob(filename))
    if matches:
        return matches[0]
    return direct


def load_audio(path: Path, sr: int = CFG.sample_rate) -> np.ndarray:
    try:
        y, _ = librosa.load(str(path), sr=sr, mono=True)
    except Exception as e:
        print(f"Could not read {path}: {e}")
        y = np.zeros(CFG.audio_len, dtype=np.float32)
    return y.astype(np.float32)


def crop_or_pad(y: np.ndarray, length: int, random_crop: bool = True) -> np.ndarray:
    if len(y) < length:
        return np.pad(y, (0, length - len(y)), mode="constant").astype(np.float32)
    if len(y) > length:
        if random_crop:
            start = np.random.randint(0, len(y) - length + 1)
        else:
            start = max(0, (len(y) - length) // 2)
        y = y[start:start + length]
    return y.astype(np.float32)


def get_shifted_chunk(y: np.ndarray, start_sec: int, end_sec: int, shift_sec: float) -> np.ndarray:
    shifted_start = start_sec + shift_sec
    shifted_end = end_sec + shift_sec
    start_sample = int(shifted_start * CFG.sample_rate)
    end_sample = int(shifted_end * CFG.sample_rate)

    if start_sample < 0:
        chunk = y[0:max(0, end_sample)]
    elif end_sample > len(y):
        chunk = y[start_sample:len(y)]
    else:
        chunk = y[start_sample:end_sample]

    return crop_or_pad(chunk, CFG.audio_len, random_crop=False)


def audio_to_logmel(y: np.ndarray) -> torch.Tensor:
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=CFG.sample_rate,
        n_fft=CFG.n_fft,
        hop_length=CFG.hop_length,
        n_mels=CFG.n_mels,
        fmin=CFG.fmin,
        fmax=CFG.fmax,
        power=CFG.power,
    )
    logmel = librosa.power_to_db(mel, ref=np.max)

    mean = logmel.mean()
    std = logmel.std() + 1e-6
    logmel = (logmel - mean) / std

    return torch.tensor(logmel, dtype=torch.float32).unsqueeze(0)


def spec_augment(x: torch.Tensor, freq_mask_prob=0.5, time_mask_prob=0.5) -> torch.Tensor:
    """Simple SpecAugment. x shape: [1, n_mels, time]."""
    x = x.clone()
    _, n_mels, n_time = x.shape

    if np.random.rand() < freq_mask_prob:
        f = np.random.randint(6, 20)
        f0 = np.random.randint(0, max(1, n_mels - f))
        x[:, f0:f0 + f, :] = 0

    if np.random.rand() < time_mask_prob:
        t = np.random.randint(8, 28)
        t0 = np.random.randint(0, max(1, n_time - t))
        x[:, :, t0:t0 + t] = 0

    return x


def make_target(labels: List[str], label_to_idx: Dict[str, int], num_classes: int) -> np.ndarray:
    target = np.zeros(num_classes, dtype=np.float32)
    for lab in labels:
        if lab in label_to_idx:
            target[label_to_idx[lab]] = 1.0
    return target


def cfg_to_dict() -> Dict[str, object]:
    cfg = {}
    for key, value in CFG.__dict__.items():
        if key.startswith("__") or callable(value):
            continue
        cfg[key] = str(value) if isinstance(value, Path) else value
    return cfg


                                                              
                         
                                                              

def prepare_metadata() -> Tuple[pd.DataFrame, List[str], Dict[str, int]]:
    taxonomy = pd.read_csv(CFG.TAXONOMY_CSV)
    class_names = taxonomy["primary_label"].astype(str).tolist()
    label_to_idx = {label: i for i, label in enumerate(class_names)}

    train = pd.read_csv(CFG.TRAIN_CSV)
    train["primary_label"] = train["primary_label"].astype(str)
    train["secondary_labels_parsed"] = train["secondary_labels"].apply(parse_secondary_labels)

    rows = []
    for _, r in tqdm(train.iterrows(), total=len(train), desc="Preparing train_audio rows"):
        if CFG.use_secondary_labels:
            labels = [r["primary_label"]] + r["secondary_labels_parsed"]
        else:
            labels = [r["primary_label"]]
        labels = [x for x in labels if x in label_to_idx]
        if not labels:
            continue
        filename = r["filename"]
        path = find_audio_path(CFG.TRAIN_AUDIO_DIR, filename, r["primary_label"])
        rows.append({
            "source": "train_audio",
            "path": str(path),
            "primary_label": r["primary_label"],
            "labels": labels,
            "start": np.nan,
            "end": np.nan,
        })

    df = pd.DataFrame(rows)

    if CFG.use_train_soundscapes and CFG.TRAIN_SOUNDSCAPES_LABELS.exists():
        ss = pd.read_csv(CFG.TRAIN_SOUNDSCAPES_LABELS)
        ss_rows = []
        for _, r in tqdm(ss.iterrows(), total=len(ss), desc="Preparing train_soundscape rows"):
            labels = str(r["primary_label"]).split(";")
            labels = [x.strip() for x in labels if x.strip() in label_to_idx]
            if not labels:
                continue
            path = CFG.TRAIN_SOUNDSCAPES_DIR / r["filename"]
            ss_rows.append({
                "source": "train_soundscape",
                "path": str(path),
                "primary_label": labels[0],
                "labels": labels,
                "start": parse_time_to_seconds(r["start"]),
                "end": parse_time_to_seconds(r["end"]),
            })

        ss_df = pd.DataFrame(ss_rows)
        if len(ss_df):
            ss_df = pd.concat([ss_df] * CFG.train_soundscape_weight, ignore_index=True)
            df = pd.concat([df, ss_df], ignore_index=True)

    if CFG.debug:
        df = df.sample(min(len(df), CFG.debug_train_rows), random_state=CFG.seed).reset_index(drop=True)

    df["fold"] = -1
    skf = StratifiedKFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)
    y = df["primary_label"].values
    for fold, (_, val_idx) in enumerate(skf.split(df, y)):
        df.loc[val_idx, "fold"] = fold

    print("DATA_DIR:", CFG.DATA_DIR)
    print("Total training rows:", len(df))
    print("Classes:", len(class_names))
    print(df["source"].value_counts())
    return df, class_names, label_to_idx


def compute_pos_weight(train_df: pd.DataFrame, label_to_idx: Dict[str, int], num_classes: int) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.float32)
    for labels in train_df["labels"]:
        for lab in labels:
            if lab in label_to_idx:
                counts[label_to_idx[lab]] += 1

    n = len(train_df)
    pos = counts
    neg = np.maximum(n - pos, 1)
    pos = np.maximum(pos, 1)
    pos_weight = neg / pos
    pos_weight = np.clip(pos_weight, 1.0, CFG.pos_weight_clip)
    return torch.tensor(pos_weight, dtype=torch.float32)


                                                              
            
                                                              

class BirdCLEFDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_to_idx: Dict[str, int], num_classes: int, train: bool = True):
        self.df = df.reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.num_classes = num_classes
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        y = load_audio(Path(r["path"]), CFG.sample_rate)

        if r["source"] == "train_soundscape" and not pd.isna(r["start"]):
            start_sample = int(float(r["start"]) * CFG.sample_rate)
            end_sample = int(float(r["end"]) * CFG.sample_rate)
            y = y[start_sample:end_sample]
            y = crop_or_pad(y, CFG.audio_len, random_crop=False)
        else:
            y = crop_or_pad(y, CFG.audio_len, random_crop=self.train)

        if self.train:
            gain = np.random.uniform(0.75, 1.25)
            y = y * gain
            if np.random.rand() < 0.35:
                y = y + np.random.normal(0, 0.003, size=len(y)).astype(np.float32)
            if np.random.rand() < 0.25:
                shift = np.random.randint(-CFG.sample_rate // 2, CFG.sample_rate // 2)
                y = np.roll(y, shift)

        x = audio_to_logmel(y)
        if self.train:
            x = spec_augment(x)

        target = make_target(r["labels"], self.label_to_idx, self.num_classes)
        target = torch.tensor(target, dtype=torch.float32)
        return x, target


                                                              
          
                                                              

class BirdCLEFModel(nn.Module):
    def __init__(self, model_name: str, num_classes: int, pretrained: bool = True, in_chans: int = 1):
        super().__init__()
        try:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                in_chans=in_chans,
                num_classes=0,
                global_pool="avg",
            )
        except Exception:
            if not pretrained:
                raise
            warnings.warn(f"Could not load pretrained weights for {model_name}; using random init.")
            self.backbone = timm.create_model(
                model_name,
                pretrained=False,
                in_chans=in_chans,
                num_classes=0,
                global_pool="avg",
            )

        n_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(CFG.dropout),
            nn.Linear(n_features, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


                                                              
                       
                                                              

def macro_auc_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    scores = []
    for c in range(y_true.shape[1]):
        yt = y_true[:, c]
        yp = y_pred[:, c]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        try:
            scores.append(roc_auc_score(yt, yp))
        except ValueError:
            pass
    return float(np.mean(scores)) if scores else np.nan


def apply_mixup(x: torch.Tensor, y: torch.Tensor):
    if (not CFG.use_mixup) or np.random.rand() > CFG.mixup_prob:
        return x, y
    lam = np.random.beta(CFG.mixup_alpha, CFG.mixup_alpha)
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y


def train_one_epoch(model, loader, optimizer, scaler, criterion):
    model.train()
    losses = []

    pbar = tqdm(loader, desc="Train", leave=False)
    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)
        x, y = apply_mixup(x, y)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=CFG.use_amp and device.type == "cuda"):
            logits = model(x)
            loss = criterion(logits, y)

        if CFG.use_amp and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
            optimizer.step()

        losses.append(loss.item())
        pbar.set_postfix(loss=np.mean(losses))

    return float(np.mean(losses))


@torch.no_grad()
def valid_one_epoch(model, loader, criterion):
    model.eval()
    losses, preds, targets = [], [], []

    for x, y in tqdm(loader, desc="Valid", leave=False):
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        prob = torch.sigmoid(logits)

        losses.append(loss.item())
        preds.append(prob.cpu().numpy())
        targets.append(y.cpu().numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    auc = macro_auc_score(targets, preds)
    return float(np.mean(losses)), auc


def train_fold(df: pd.DataFrame, class_names: List[str], label_to_idx: Dict[str, int], fold: int):
    train_df = df[df.fold != fold].reset_index(drop=True)
    valid_df = df[df.fold == fold].reset_index(drop=True)

    train_ds = BirdCLEFDataset(train_df, label_to_idx, len(class_names), train=True)
    valid_ds = BirdCLEFDataset(valid_df, label_to_idx, len(class_names), train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=CFG.valid_batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )

    model = BirdCLEFModel(CFG.model_name, len(class_names), CFG.pretrained, CFG.in_chans).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epochs, eta_min=CFG.min_lr)

    if CFG.use_pos_weight:
        pos_weight = compute_pos_weight(train_df, label_to_idx, len(class_names)).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=CFG.use_amp and device.type == "cuda")

    best_auc = -np.inf
    best_path = CFG.OUTPUT_DIR / f"single_timm_v2_{CFG.model_name.replace('/', '_')}_fold{fold}.pth"

    for epoch in range(1, CFG.epochs + 1):
        print(f"\nFold {fold} | Epoch {epoch}/{CFG.epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, criterion)
        valid_loss, valid_auc = valid_one_epoch(model, valid_loader, criterion)
        scheduler.step()

        print(f"fold={fold} train_loss={train_loss:.5f} valid_loss={valid_loss:.5f} valid_macro_auc={valid_auc:.5f}")

        if np.isfinite(valid_auc) and valid_auc > best_auc:
            best_auc = valid_auc
            torch.save({
                "model": model.state_dict(),
                "class_names": class_names,
                "cfg": cfg_to_dict(),
                "fold": fold,
                "valid_auc": float(valid_auc),
            }, best_path)
            print("Saved:", best_path)

    print(f"Fold {fold} best AUC: {best_auc}")
    return best_path


def train_all_folds():
    df, class_names, label_to_idx = prepare_metadata()
    saved_paths = []
    for fold in CFG.train_folds:
        saved_paths.append(train_fold(df, class_names, label_to_idx, fold=fold))
    return saved_paths, class_names


                                                              
                        
                                                              

@torch.no_grad()
def predict_batch(models: List[nn.Module], waves: List[np.ndarray]) -> np.ndarray:
    xs = [audio_to_logmel(w) for w in waves]
    x = torch.stack(xs).to(device)

    all_probs = []
    for model in models:
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)

    return np.mean(all_probs, axis=0)


def discover_model_paths() -> List[Path]:
    local = sorted(CFG.OUTPUT_DIR.glob("*.pth"))
    if local:
        return local
    kaggle = sorted(Path("/kaggle/input").rglob("*.pth"))
    return kaggle


def load_models(model_paths: List[Path]):
    models = []
    class_names = None

    for p in model_paths:
        print("Loading model:", p)
        ckpt = torch.load(p, map_location="cpu")
        if class_names is None:
            class_names = ckpt["class_names"]
        else:
            assert class_names == ckpt["class_names"], "Class mismatch between checkpoints"

        cfg = ckpt.get("cfg", {})
        model_name = cfg.get("model_name", CFG.model_name)

        model = BirdCLEFModel(model_name, len(class_names), pretrained=False, in_chans=CFG.in_chans).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        models.append(model)

    return models, class_names


def make_submission(model_paths: Optional[List[Path]] = None):
    sample = pd.read_csv(CFG.SAMPLE_SUBMISSION)
    sample_cols = sample.columns.tolist()

    if model_paths is None:
        model_paths = discover_model_paths()

    assert len(model_paths) > 0, "No model checkpoints found"
    models, class_names = load_models(model_paths)

    test_files = sorted(list(CFG.TEST_SOUNDSCAPES_DIR.glob("*.ogg")))
    print("Number of test files:", len(test_files))

    if len(test_files) == 0:
        print("No files found in test_soundscapes. Writing dummy submission.")
        sample.to_csv("submission.csv", index=False)
        return sample

    row_predictions = {}

    for shift in CFG.tta_shifts:
        print(f"\nTTA shift: {shift}")
        batch_waves, batch_row_ids = [], []

        for path in tqdm(test_files, desc=f"Inference shift {shift}"):
            y = load_audio(path, CFG.sample_rate)
            stem = path.stem

            for end_sec in range(5, 65, 5):
                start_sec = end_sec - 5
                chunk = get_shifted_chunk(y, start_sec, end_sec, shift)
                row_id = f"{stem}_{end_sec}"

                batch_waves.append(chunk)
                batch_row_ids.append(row_id)

                if len(batch_waves) >= CFG.infer_batch_size:
                    probs = predict_batch(models, batch_waves)
                    for rid, p in zip(batch_row_ids, probs):
                        row_predictions.setdefault(rid, []).append(p)
                    batch_waves, batch_row_ids = [], []

        if batch_waves:
            probs = predict_batch(models, batch_waves)
            for rid, p in zip(batch_row_ids, probs):
                row_predictions.setdefault(rid, []).append(p)
            batch_waves, batch_row_ids = [], []

    rows = []
    for rid, preds in tqdm(row_predictions.items(), desc="Averaging TTA"):
        p = np.mean(preds, axis=0)
        row = {"row_id": rid}
        row.update({cls: float(p[i]) for i, cls in enumerate(class_names)})
        rows.append(row)

    sub = pd.DataFrame(rows)

    for col in sample_cols:
        if col not in sub.columns:
            sub[col] = 0.0
    sub = sub[sample_cols]

    sub.to_csv("submission.csv", index=False)
    print("Saved submission.csv", sub.shape)
    return sub


                                                              
          
                                                              

if __name__ == "__main__":
    print("Device:", device)
    print("DATA_DIR:", CFG.DATA_DIR)
    print("OUTPUT_DIR:", CFG.OUTPUT_DIR)

    saved_paths = []
    if CFG.RUN_TRAIN:
        saved_paths, _ = train_all_folds()

    if CFG.RUN_INFER:
        make_submission(saved_paths if saved_paths else None)
