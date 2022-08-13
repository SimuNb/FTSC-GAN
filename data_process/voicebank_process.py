import glob
import os
import sys
import random
import tqdm
import librosa
import numpy as np


def load_and_trim(path, sr):
    # 去掉两端的静音
    audio, sr = librosa.load(path, sr=sr)  # 加载音频
    energy = librosa.feature.rms(audio)  # 计算能量
    frames = np.nonzero(energy >= np.max(energy) / 5)  # 如果能量大则保留 说明这段不是两端的静音
    indices = librosa.core.frames_to_samples(frames)[1]
    audio = audio[indices[0]:indices[-1]] if indices.size else audio[0:0]

    return audio, sr  # 去掉两端静音的音频和采样频率



def SNR(x1, x2):
    from numpy.linalg import norm
    return 20 * np.log10(norm(x1) / norm(x2))

def voice_normalize(data):
    return (data - np.mean(data)) / np.max(np.abs(data))

def signal_by_db(speech, noise, snr):
    sum_s = np.sum(speech ** 2)
    sum_n = np.sum(noise ** 2)
    # 信噪比为-5dB时的权重,snr=-5
    x = np.sqrt(sum_s / (sum_n * pow(10, snr/10)))
    noise = x * noise
    mix = speech + noise
    return mix, speech, noise


# 统计最后一帧补零的位数
def padding_zero_num(num_list):
    num = 0
    for i in range(len(num_list)):
        if num_list[len(num_list) - 1 - i] == 0:
            num += 1
        else:
            break
    return num


def enframe(signal, frame_len, frame_stride):
    '''
    将语音信号转化为帧
    '''
    signal_length = len(signal)  # 信号总长度
    if signal_length <= frame_len:      # 如果信号长度小于一帧的长度，则帧数定义为1
        frame_num = 1                    # frame_num表示帧数量
    else:
        frame_num = int(np.ceil((1.0 * signal_length - frame_len + frame_stride) / frame_stride))  # 处理后，所有帧的数量，不要最后一帧
    pad_length = int((frame_num - 1) * frame_stride + frame_len)                   # 所有帧加起来总的平铺后的长度
    pad_signal = np.pad(signal, (0, pad_length - signal_length), 'constant')     # 0填充最后一帧
    indices = np.tile(np.arange(0, frame_len), (frame_num, 1)) + np.tile(np.arange(0, frame_num * frame_stride, frame_stride), (frame_len, 1)).T  # 每帧的索引
    indices = np.array(indices, dtype=np.int32)
    frames = pad_signal[indices]

    return frames


def get_stop_point(frame):
    stop_point = 0
    for i in range(len(frame.tolist())):
        item = frame[-i]
        if padding_zero_num(item.tolist()) >= 15384:
            stop_point = i
        else:
            break
    return stop_point


def sav_npy(frame, sav_name, stop_point=0):
    if os.path.exists(sav_name):
        os.remove(sav_name)
    sav_list = frame[:(len(frame.tolist()) - stop_point)]
    np.save(sav_name, sav_list)

def my_glob(speech_path):
    speech_list = []
    for dir, sub_dir, file in os.walk(speech_path):
        for item in file:
            speech_list.append(dir + "\\" + item)
    return speech_list


def main():
    frame_len = 16384
    frame_strid = 16384
    sr = 16000
    train_path = r"G:\paper\common_se_data\noisy_testset_wav"
    train_data_list = glob.glob(train_path + "\\*")


    mix_npy_path = r"G:\paper\common_se_data\voicebank_train\test\mix"
    speech_npy_path = r"G:\paper\common_se_data\voicebank_train\test\speech"

    for noisy in tqdm.tqdm(train_data_list):


        clean = noisy.replace("noisy_", "clean_")

        noisy_y, sr = librosa.load(noisy, sr)
        clean_y, sr = librosa.load(clean, sr)

        mix_name = noisy.split("\\")[-1].replace(".wav", "") + "_" + str(len(noisy_y)) + "_mix.npy"
        speech_name = noisy.split("\\")[-1].replace(".wav", "") + "_" + str(len(clean_y)) + "_speech.npy"

        mix_frame = enframe(noisy_y, frame_len, frame_strid)
        speech_frame = enframe(clean_y, frame_len, frame_strid)

        np.save(mix_npy_path + "\\" + mix_name, mix_frame)
        np.save(speech_npy_path + "\\" + speech_name, speech_frame)











if __name__ == "__main__":
    main()
