import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os
import csv
import torch
import numpy as np
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import librosa
import soundfile
import audioread.ffdec

from simai_tokenizer import tokenize_chart, build_vocab

def load_audio_wave(sr, max_duration, audio_path, fallback_load_method=None):
    if fallback_load_method is None or len(fallback_load_method) == 0:
        raise ValueError(f"Cannot load: {audio_path}, {os.path.exists(audio_path)}")
    try:
        audio = fallback_load_method[0](audio_path)
        y, sr = librosa.load(audio, sr=sr, duration=max_duration)
        if len(y) == 0:
            raise ValueError("")
        return y, sr
    except:
        return load_audio_wave(sr, max_duration, audio_path, fallback_load_method[1:])

def load_audio_without_cache(audio_path, n_mels, audio_hop_length, n_fft, sr, max_duration):
    y, sr = load_audio_wave(sr, max_duration, audio_path, [
        audioread.ffdec.FFmpegAudioFile,
        soundfile.SoundFile, 
        lambda x: x
    ])
    y = librosa.feature.melspectrogram(y=y, sr=sr,
                                       n_mels=n_mels,
                                       hop_length=audio_hop_length,
                                       n_fft=n_fft)
    y = np.log1p(y).astype(np.float16)
    return y

def load_audio(cache_dir, audio_path, n_mels, audio_hop_length, n_fft, sr, max_duration):
    audio_path = audio_path.strip()
    if cache_dir is None:
        return load_audio_without_cache(audio_path, n_mels, audio_hop_length, n_fft, sr, max_duration)
    cache_name = f"{os.path.basename(os.path.dirname(audio_path))}-{os.path.basename(audio_path)}.npz"
    cache_path = os.path.join(cache_dir, cache_name)
    if os.path.isfile(cache_path):
        return np.load(cache_path)['y']
    
    os.makedirs(cache_dir, exist_ok=True)
    y = load_audio_without_cache(audio_path, n_mels, audio_hop_length, n_fft, sr, max_duration)
    np.savez_compressed(cache_path, y=y)
    return y

class MaiGenDataset(Dataset):
    def __init__(self, data_dir, csv_file, cache_dir, 
                 sr=22050, n_fft=512, n_mels=128, max_audio_frame=32768, 
                 audio_hop_length=None):
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.sr = sr
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.max_audio_frame = max_audio_frame
        
        # Consistent with Mug-Diffusion
        self.audio_hop_length = audio_hop_length if audio_hop_length else n_fft // 4
        self.max_duration = (self.audio_hop_length / sr) * max_audio_frame

        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            self.data = [row for row in reader]

        self.vocab = build_vocab()
        self.bos_idx = self.vocab['<bos>']
        self.eos_idx = self.vocab['<eos>']
        self.pad_idx = self.vocab['<pad>']
        self.unk_idx = self.vocab['<unk>']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        path = os.path.join(self.data_dir, item['path'])
        chart_path = os.path.join(path, "maidata.txt")
        audio_path = os.path.join(path, "track.mp3")
        diff = item['diff']

        try:
            # Load audio
            audio = load_audio(
                self.cache_dir, audio_path, 
                self.n_mels, self.audio_hop_length, 
                self.n_fft, self.sr, self.max_duration
            ).astype(np.float32)

            # Pad or truncate audio
            t = audio.shape[1]
            if t < self.max_audio_frame:
                audio = np.concatenate([
                    audio,
                    np.zeros((self.n_mels, self.max_audio_frame - t), dtype=np.float32)
                ], axis=1)
            elif t > self.max_audio_frame:
                audio = audio[:, :self.max_audio_frame]

            # Tokenize chart
            str_tokens = tokenize_chart(chart_path, diff)
            if str_tokens is None:
                raise ValueError(f"Failed to tokenize {chart_path} at diff {diff}")

            token_ids = [self.bos_idx]
            for tok in str_tokens:
                token_ids.append(self.vocab.get(tok, self.unk_idx))
            token_ids.append(self.eos_idx)

            return {
                "audio": torch.tensor(audio, dtype=torch.float32),
                "tokens": torch.tensor(token_ids, dtype=torch.long)
            }
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Return empty/dummy item, collate_fn will filter it
            return None

def mai_gen_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return {}

    audios = torch.stack([b['audio'] for b in batch])
    
    # Pad tokens
    tokens_list = [b['tokens'] for b in batch]
    max_len = max(len(t) for t in tokens_list)
    
    # Assume 0 is <pad>
    padded_tokens = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, t in enumerate(tokens_list):
        padded_tokens[i, :len(t)] = t

    return {
        "audio": audios,
        "tokens": padded_tokens
    }

class MaiGenDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, csv_file, cache_dir, batch_size=4, num_workers=4, 
                 sr=22050, n_fft=512, n_mels=128, max_audio_frame=32768):
        super().__init__()
        self.data_dir = data_dir
        self.csv_file = csv_file
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sr = sr
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.max_audio_frame = max_audio_frame

    def setup(self, stage=None):
        full_dataset = MaiGenDataset(
            self.data_dir, self.csv_file, self.cache_dir,
            self.sr, self.n_fft, self.n_mels, self.max_audio_frame
        )
        
        # Simple 90/10 split
        train_len = int(0.9 * len(full_dataset))
        val_len = len(full_dataset) - train_len
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_dataset, [train_len, val_len],
            generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, 
            shuffle=True, num_workers=self.num_workers,
            collate_fn=mai_gen_collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers,
            collate_fn=mai_gen_collate_fn
        )
