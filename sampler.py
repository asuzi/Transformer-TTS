import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import random
from hyperparams import hp
from utils import collateFn

bucket_size_text = 60
bucket_size_mel = 300


class DualBucketBatchSampler(Sampler):
    def __init__(self, text_lengths, mel_lengths, batch_size,
                 bucket_size_text=bucket_size_text, bucket_size_mel=bucket_size_mel,
                 shuffle=True, drop_last=True):
        assert len(text_lengths) == len(mel_lengths)
        self.text_lengths = text_lengths
        self.mel_lengths = mel_lengths
        self.batch_size = batch_size
        self.bucket_size_text = bucket_size_text
        self.bucket_size_mel = bucket_size_mel
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Bucket samples by (text_bucket, mel_bucket)
        self.buckets = defaultdict(list)
        for idx, (t_len, m_len) in enumerate(zip(text_lengths, mel_lengths)):
            text_bucket = t_len // bucket_size_text
            mel_bucket = m_len // bucket_size_mel
            self.buckets[(text_bucket, mel_bucket)].append(idx)

        self.batches = self._create_batches()

    def _create_batches(self):
        batches = []
        for bucket in self.buckets.values():
            if self.shuffle:
                random.shuffle(bucket)
            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i:i + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                batches.append(batch)
        if self.shuffle:
            random.shuffle(batches)
        return batches

    def __iter__(self):
        if self.shuffle:
            self.batches = self._create_batches()
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def collateFnFixedPad(batch, bucket_size_text=bucket_size_text, bucket_size_mel=bucket_size_mel):
    # Extract sequences
    texts = [item[0] for item in batch]
    mels = [item[1].squeeze(0).transpose(0, 1) for item in batch]
    mel_lens = torch.LongTensor([item[2] for item in batch])

    # Get max lengths in this batch
    max_text_len = max(len(t) for t in texts)
    max_mel_len = max(mel_lens).item()

    # Find padding sizes: round *up* to nearest bucket
    pad_text_len = ((max_text_len - 1) // bucket_size_text + 1) * bucket_size_text
    pad_mel_len = ((max_mel_len - 1) // bucket_size_mel + 1) * bucket_size_mel

    if pad_text_len > 175:
        pad_text_len = 175

    if pad_mel_len > 866:
        pad_mel_len = 866

    # Pad text
    padded_texts = torch.zeros(len(texts), pad_text_len, dtype=torch.long)
    for i, text in enumerate(texts):
        padded_texts[i, :text.size(0)] = text

    # Pad mels
    padded_mels = torch.zeros(len(mels), pad_mel_len, mels[0].size(1), dtype=torch.float)
    for i, mel in enumerate(mels):
        padded_mels[i, :mel.size(0), :] = mel

    return padded_texts, padded_mels, mel_lens



