import os
import argparse
import torch
import numpy as np
import yaml

from models.seq2seq import ChartGenerator
from data.dataset import load_audio_without_cache
from simai_tokenizer import build_vocab, save_chart

def main():
    parser = argparse.ArgumentParser(description="Generate Maimai chart from audio.")
    parser.add_argument("audio_path", help="Path to input audio file")
    parser.add_argument("--checkpoint", "-c", required=True, help="Path to model checkpoint (.ckpt)")
    parser.add_argument("--config", help="Path to training config.yaml (optional, useful for audio params)", default="configs/train.yaml")
    parser.add_argument("--output", "-o", default="output_chart.txt", help="Output chart file path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    args = parser.parse_args()

    # Load config for audio parameters
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    data_args = config['data']
    sr = data_args.get('sr', 22050)
    n_fft = data_args.get('n_fft', 512)
    n_mels = data_args.get('n_mels', 128)
    max_audio_frame = data_args.get('max_audio_frame', 32768)
    audio_hop_length = n_fft // 4
    max_duration = (audio_hop_length / sr) * max_audio_frame

    print(f"Loading audio from {args.audio_path}...")
    # Load and process audio
    mel = load_audio_without_cache(args.audio_path, n_mels, audio_hop_length, n_fft, sr, max_duration)
    
    # Pad or truncate (same as dataset)
    t = mel.shape[1]
    if t < max_audio_frame:
        mel = np.concatenate([mel, np.zeros((n_mels, max_audio_frame - t), dtype=np.float32)], axis=1)
    elif t > max_audio_frame:
        mel = mel[:, :max_audio_frame]

    mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).to(args.device)

    print(f"Loading model from {args.checkpoint}...")
    model = ChartGenerator.load_from_checkpoint(args.checkpoint, map_location=args.device)
    model.eval()

    vocab = build_vocab()
    idx_to_token = {v: k for k, v in vocab.items()}
    bos_idx = vocab['<bos>']
    eos_idx = vocab['<eos>']

    print("Generating chart tokens...")
    # Generate tokens
    predicted_ids = model.generate(mel_tensor, bos_idx, eos_idx)
    predicted_ids = predicted_ids[0].cpu().numpy() # batch size 1

    # Convert to string tokens
    tokens = []
    for idx in predicted_ids:
        if idx == eos_idx:
            break
        if idx != bos_idx:
            tokens.append(idx_to_token.get(idx, '<unk>'))

    print(f"Generated {len(tokens)} tokens. Saving to {args.output}...")
    save_chart(tokens, args.output)

if __name__ == "__main__":
    main()
