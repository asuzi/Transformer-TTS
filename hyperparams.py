import torch



class Hyperparams:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    continue_training_from_checkpoint = True
    save_latest = True  # True: Saves the latest epoch | False: Saves the lowest test-loss model state.

    start_frame_mel_fill_value = -0.1

    # ======[ Common Model ]====== 
    gradient_accumulation_steps = 4
    batch_size = 16 # 16 # 8
    n_embd = 256 # 384 # 512
    forward_expansion = 4
    lr = 1.0 
    epoch = 500 

    entropy_lambda = 2e-3
    sigma = 0.4
    gamma = 1.0

    attn_temperature = 0.7
    lr_noam = 1.0
    warmup_steps = 8000 

    TF_duration = 80000
    min_p_teacher = 0.2
    decay_duration = 160000 - (TF_duration + warmup_steps)

    p_continue_min = 0.4   # minimal continuation early
    p_continue_max = 0.95   # max continuation late

    # ======[ Encoder ]====== 
    enc_n_head = 4
    enc_n_block = 4
    enc_dropout = 0.1

    # ======[ Decoder ]====== 
    dec_n_head = 4
    dec_n_block = 4
    dec_dropout = 0.1
    dec_forward_expansion = 4

    # ======[ PostNet ]====== 
    postnet_n_embd = 256    
    postnet_layers = 5      
    postnet_dropout = 0.5   
    postnet_kernel = 5      
    postnet_stride = 1      
    postnet_dilation = 1    


    # ======[ init ]======
    train_test_split = 1.0 # 0.8
    
    text_vocab_len = 91
    ph_vocab_len = 80

    text_sequence_len = 175 # longest   -  224 for non phonemes
    mel_sequence_len = 866 # longest

    max_pos_len = int(text_sequence_len + mel_sequence_len)


    # ======[ Paths ]====== 
    TRAIN_DATASET_PATH = f"V:\\TTS_actual\\training_data\\processed_data\\trainLoader.pth"
    TEST_DATASET_PATH = f"V:\\TTS_actual\\training_data\\processed_data\\testLoader.pth"
    VOCAB_PATH = "V:\\TTS_actual\\training_data\\processed_data\\vocab.json"

    WAV_ROOT = "V:\\TTS_actual\\training_data\\raw_data\\LJSpeech-1.1\\wavs\\"
    METADATA_CSV = "V:\\TTS_actual\\training_data\\raw_data\\LJSpeech-1.1\\metadata.csv"
    SAVE_ROOT = "V:\\TTS_actual\\training_data\\processed_data\\"
    MODEL_SAVE_ROOT = "V:\\TTS_actual\\model_save_folder\\"
    WAV_SAVE_ROOT = "V:\\TTS_actual\\wav_save_folder\\"
    PLOTTING_SAVE_ROOT = "V:\\TTS_actual\\plotting_save_folder\\"

   
    kernel_size = 5
    stride = 1
    dilation = 1


    # ======[ Mel-Spectrogram ]====== 
    n_mel_bin = 80

    sample_rate = 22050
    n_fft = 1024
    n_stft = int((n_fft//2)+1)
    hop_length = 256
    win_length = 1024
    max_db = 100
    power = 1.0
    f_min = 0.0
    f_max = 8000.0


hp = Hyperparams






