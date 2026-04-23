import torch
import torchaudio
from torchaudio import transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from hyperparams import hp
import os
import re
import IPython
from g2p_en import G2p
import json
import librosa


g2p = G2p()



class TextMelDataset(Dataset):
    def __init__(self, texts, text_len, mel_spectrograms, mel_len):
        self.texts = texts
        self.text_len = text_len
        self.mel_spectrograms = mel_spectrograms
        self.mel_len = mel_len

    def __len__(self):
        if len(self.texts) == len(self.mel_spectrograms) and len(self.mel_len) == len(self.text_len):
            return len(self.texts)
        else:
            print("critical, length missmatch!")
            return None

    def __getitem__(self, idx):
        text = self.texts[idx]
        text_len = self.text_len[idx]
        mel_spec = self.mel_spectrograms[idx]
        mel_len = self.mel_len[idx]

        return text, text_len, mel_spec, mel_len


class CreateDataset():
    def __init__(self):


        self.text_list = [] # Plain text
        self.ph_list = [] # Plain phenomenoms

        self.text_vocab = set() # Vocab for plain text
        self.ph_vocab = set() # Vocab for phenomenoms


        self.wav_list = [] # Names of audio files (.waf)
        self.mel_list = [] # List of audio files transformed into PowerDB MEL-spectrograms as Tensor

        self.text_tokenized = [] # List of PyTorch tensors of plain text
        self.ph_tokenized = []  # List of PyTorch tensors of plain phenomenoms


    def readData(self, csv_path):
        # Read CSV and get all of the data. Split data into a list. Loop list and append to own lists.
        with open(csv_path, "r", encoding="UTF-8") as csv:
            t = csv.read()
            data = re.split('\n|\|', t)
            csv.close

        # Append .wav filenames to self.wav_list
        for i in range(0,len(data),3):
            self.wav_list.append(data[i])

        # Append non-normalized text
        for i in range(1,len(data),3):
            phenomenoms = g2p(data[i])


            self.text_list.append(data[i])
            self.ph_list.append(phenomenoms)

            self.text_vocab.update(data[i])
            self.ph_vocab.update(phenomenoms)
        

    def transformData(self):

        self.tokenizer = Tokenizer(text_vocab=self.text_vocab, ph_vocab=self.ph_vocab)



        # Turn all .wav files into powerDB mel-spectrograms and save the tensors in a list.
        for filename in self.wav_list:
            data = wav_to_hifigan_mel(hp.WAV_ROOT + filename + ".wav")
            self.mel_list.append(data)

        # Tokenize plain text list
        for letter in self.text_list:
            token_array = self.tokenizer.encode_text(text=letter)
            token_tensor = torch.Tensor(token_array)
            self.text_tokenized.append(token_tensor)

        # Tokenize plain phenomenom list
        for ph in self.ph_list:
            token_array = self.tokenizer.encode_phonemes(ph)
            token_tensor = torch.Tensor(token_array)
            self.ph_tokenized.append(token_tensor)


    def generateDataset(self, max_text_len, max_mel_len):
        print("Generating new dataset! Starting CreateDataset.readData()")
        self.readData(hp.METADATA_CSV)

        print("Starting CreateDataset.transformData()")
        self.transformData()

        # ADD PADDING TO THE DATA.
        print("PADDING DATA....")



        self.mel_lengths = [mel.shape[-1] for mel in self.mel_list]
        longest_mel = max(self.mel_lengths)

        self.text_lengths = [ph.shape[-1] for ph in self.ph_tokenized]
        longest_text = max(self.text_lengths)




        print(f"Longest text sequence vs hp.text_sequence_len | {longest_text} vs {hp.text_sequence_len}")
        print(f"Longest mel sequence vs hp.mel_sequence_len | {longest_mel} vs {hp.mel_sequence_len}")

        if longest_text > max_text_len:
            print(f"[ERROR]> Text sequences are longer than max_text_len!! Automatically setting max_text_len to {longest_text}. You need to update the new value in hyperparams.py to avoid further errors!")
            max_text_len = longest_text

        if longest_mel > max_mel_len:
            print(f"[ERROR]> Mel sequences are longer than max_mel_len!! Automatically setting max_mel_len to {longest_mel}. You need to update the new value in hyperparams.py to avoid further errors!")
            max_mel_len = longest_mel

  #      text = self.addPadding(data=self.ph_tokenized,
 #                              target_len=max_text_len)
        
 #       mel = self.addPadding(data=self.mel_list,
 #                             target_len=max_mel_len)

        # if using buckets/variable padding
        text = self.ph_tokenized
        mel = self.mel_list

        # SPLIT DATA TO TRAIN AND TEST SETS.
        print("SPLITTING DATA...")
        cut = int(hp.train_test_split * (len(text)))

        text_train = text[:cut]
        text_test = text[cut:]
        text_len_train = self.text_lengths[:cut]
        text_len_test = self.text_lengths[cut:]

        mel_train = mel[:cut]
        mel_test = mel[cut:]
        mel_len_train = self.mel_lengths[:cut]
        mel_len_test = self.mel_lengths[cut:]

        print(len(text_train))
        print(len(text_test))
        print()
        print(len(mel_train))
        print(len(mel_test))
        print()
        print(len(mel_len_train))
        print(len(mel_len_test))
        print()

        train_dataset = TextMelDataset(texts=text_train, text_len=text_len_train, mel_spectrograms=mel_train, mel_len=mel_len_train)
        test_dataset = TextMelDataset(texts=text_test, text_len=text_len_test, mel_spectrograms=mel_test, mel_len=mel_len_test)

        print("SAVING DATALOADERS TO: "+ hp.SAVE_ROOT)
        torch.save(train_dataset, f"{hp.SAVE_ROOT}trainLoader.pth")
        torch.save(test_dataset, f"{hp.SAVE_ROOT}testLoader.pth")
        with open(f"{hp.SAVE_ROOT}vocab.json", 'w') as v:
            json.dump(list(self.ph_vocab), v)
            v.close()
        print("Finished!.")

    def addPadding(self, data, target_len):
        result = []
        for i in data:
            padding_size = target_len - i.shape[-1]
            padded = F.pad(i, (0, padding_size))
            result.append(padded)

        return result



class Tokenizer:
    def __init__(self, text_vocab:set=None, ph_vocab:set=None):

        self.text_vocab = text_vocab if text_vocab else set()
        self.ph_vocab = ph_vocab if ph_vocab else set()

        self.special_tokens = ['PADDING', 'SOS', 'EOS']

        for token in self.special_tokens:
            self.text_vocab.add(token)
            self.ph_vocab.add(token)


        self.text_token_to_id, self.text_id_to_token = self._map_vocab(self.text_vocab)
        self.ph_token_to_id, self.ph_id_to_token = self._map_vocab(self.ph_vocab)

  #      print(f"The length of plain text vocab is: {len(self.text_token_to_id)}")
 #       print(f"The length of phoneme vocab is: {len(self.ph_token_to_id)}")


    def _map_vocab(self, vocab:set):
        vocab_list = ['PADDING'] + sorted(token for token in vocab if token != 'PADDING')
        token_to_id = {token: idx for idx, token in enumerate(vocab_list)}        
        id_to_token = {idx: token for token, idx in token_to_id.items()}

        return token_to_id, id_to_token
    

    def encode_text(self, text):
        tokens = list(text)
        return [self.text_token_to_id.get(tok, self.text_token_to_id['PADDING']) for tok in tokens]

    def decode_text(self, token_ids):
        return ''.join(self.text_id_to_token.get(idx, '-') for idx in token_ids)


    def encode_phonemes(self, phoneme_list):
        tokens = ['SOS'] + phoneme_list + ['EOS']
        return [self.ph_token_to_id.get(tok, self.ph_token_to_id['PADDING']) for tok in tokens]

    def decode_phonemes(self, token_ids):
        tokens = [self.ph_id_to_token.get(idx, '-') for idx in token_ids]
        if tokens and tokens[0] == 'SOS':
            tokens = tokens[1:]
        if tokens and tokens[-1] == 'EOS':
            tokens = tokens[:-1]
        return tokens
    

def encode_tokens(s, vocab):
    string_to_int = { ch:i for i, ch in enumerate(vocab) }
    return [string_to_int.get(c, 0) for c in s]

def decode_tokens(l, vocab):
    int_to_string = { i:ch for i, ch in enumerate(vocab) }
    return "".join([int_to_string.get(i, '') for i in l])



_mel_basis = {}
_hann_window = {}

def hifigan_mel_spectrogram(y, center, device="cpu"):

    global _mel_basis, _hann_window

    mel_key = f"{hp.f_max}_{device}"
    if mel_key not in _mel_basis:
        mel = librosa.filters.mel(
            sr=hp.sample_rate,
            n_fft=hp.n_fft,
            n_mels=hp.n_mel_bin,
            fmin=hp.f_min,
            fmax=hp.f_max
        )
        mel = torch.from_numpy(mel).float().to(device)
        _mel_basis[mel_key] = mel

        _hann_window[device] = torch.hann_window(hp.win_length).to(device)

    mel_basis = _mel_basis[mel_key]
    hann_window = _hann_window[device]

 #   y = torch.nn.functional.pad(y.unsqueeze(1), (int((hp.n_fft - hp.hop_length) / 2), int((hp.n_fft - hp.hop_length) / 2)), mode='reflect')
 #   y = y.squeeze(1)

    spec = torch.stft(y, hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length, window=hann_window, center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
    spec = torch.abs(spec)
    mel_spec = torch.matmul(mel_basis, spec)
    mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))

    return mel_spec

def wav_to_hifigan_mel(wav_path):
    wav, sample_rate = torchaudio.load(wav_path)
    if sample_rate != hp.sample_rate:
        raise ValueError(f"Sample rate mismatch. Expected 22050, got {sample_rate}.")

    # Make sure wav is 1D or batch dimension 1
    if wav.dim() > 1 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)  # convert to mono

    mel = hifigan_mel_spectrogram(
        wav,
        device=wav.device,
        center=False
    )
    return mel  # shape: [1, 80, T]

def wav_to_mel_log(wav_path):
    wav, sample_rate = torchaudio.load(wav_path, normalize=True)
    if sample_rate != hp.sample_rate:
        raise ValueError(f"Expected {hp.sample_rate}, got {sample_rate}.")
    
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=hp.sample_rate,
        n_fft=hp.n_fft,
        win_length=hp.win_length,
        hop_length=hp.hop_length,
        n_mels=hp.n_mel_bin,
        power=hp.power,
        f_min=hp.f_min,
        f_max=hp.f_max,
        mel_scale='slaney', 
        norm=None,
        center=False
    )

    mel = mel_transform(wav)                # [1, 80, T]
    mel_log = torch.log(torch.clamp(mel, min=1e-5))  # log-mel
    
    return mel_log



def mel_log_to_wav_griffinLim(mel_log : torch.Tensor, save_path):
    inverse_mel = torchaudio.transforms.InverseMelScale(
        n_stft=hp.n_stft,
        n_mels=hp.n_mel_bin,
        sample_rate=hp.sample_rate,
    ).to(hp.device)

    mel = torch.exp(mel_log) # de-log
    linear = inverse_mel(mel).to(hp.device)

    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=hp.n_fft,
        win_length=hp.win_length,
        hop_length=hp.hop_length,
        n_iter=200,
        power=hp.power
    ).to(hp.device)

    wav = griffin_lim(linear)
    wav = wav / wav.abs().max() * 0.95 # Add smoothing

    wav = wav.cpu()
    torchaudio.save(save_path, wav, sample_rate=hp.sample_rate)
    print(f"Saved audio to: {save_path}")

    return wav


def pseu_wav_to_file(pseu_wav, savename):
    audio = IPython.display.Audio(pseu_wav.detach().cpu().numpy(), rate=hp.sample_rate)
    with open(f"{hp.WAV_SAVE_ROOT}{savename}.wav", 'wb') as f:
        f.write(audio.data)   
        f.close()
    
    print(f"New audio file created at {hp.WAV_SAVE_ROOT}{savename}.wav")



def collateFn(batch):
    # batch is a list of tuples. (text = torch.tensor) (text_len = int) (mel = torch.tensor) (mel_len = int)



    mels = [item[2] for item in batch]
    mels = [mel.squeeze(0).transpose(0, 1) for mel in mels]
    mels = pad_sequence(mels, batch_first=True, padding_value=0.0)  # Float tensor

    texts = [item[0] for item in batch]
    texts = pad_sequence(texts, batch_first=True, padding_value=0)  # Int tensor

    mel_lens = [item[3] for item in batch]
    mel_lens = torch.LongTensor(mel_lens)

    text_lens = [item[1] for item in batch]
    text_lens = torch.LongTensor(text_lens)

    return texts, text_lens, mels, mel_lens






















