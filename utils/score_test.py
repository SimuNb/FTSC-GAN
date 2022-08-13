import glob
import tqdm
import numpy as np
import librosa
import warnings
from pypesq import pesq as pypesq
from pystoi import stoi as pystoi
from pysepm.qualityMeasures import composite, SNRseg
warnings.filterwarnings("ignore")  # 忽略告警

new_dict = {"pesq":{-2.5:{"val":0,"nums":0}, 2.5:{"val":0,"nums":0}, 7.5:{"val":0,"nums":0}, 12.5:{"val":0,"nums":0}},
            "stoi":{-2.5:{"val":0,"nums":0}, 2.5:{"val":0,"nums":0}, 7.5:{"val":0,"nums":0}, 12.5:{"val":0,"nums":0}},
            "ssnr":{-2.5:{"val":0,"nums":0}, 2.5:{"val":0,"nums":0}, 7.5:{"val":0,"nums":0}, 12.5:{"val":0,"nums":0}}}

def get_score():

    speech_wav = r"G:\paper\common_se_data\noisy_testset_wav"
    speech_wav_list = glob.glob(speech_wav + "\*")

    snrseg_list = []
    pesq_list = []
    stoi_list = []
    old_stoi_list = []
    old_snrseg_list = []
    old_pesq_list = []

    sr = 16000

    for i in tqdm.tqdm(range(len(speech_wav_list))):
        item = speech_wav_list[i]

        gen_speech = item.replace("noisy_", "clean_wav")
        mix_speech = item.replace("speech_wav", "mix_wav")




        speech_y = np.load(item)
        gen_speech_y, _ = librosa.load(gen_speech, sr)
        mix_speech_y, _ = librosa.load(mix_speech, sr)
        snr = int(item.split("/")[-1].split("_")[0][:-2])

        if snr < 0:
            snr = snr - 0.5
        else:
            snr = snr + 0.5

        if len(speech_y) == len(gen_speech_y) == len(mix_speech_y):


            pesq_score = pypesq(speech_y, gen_speech_y, sr)
            stoi_score = pystoi(speech_y, gen_speech_y, sr)
            snrseg_score = SNRseg(speech_y, gen_speech_y, fs=sr)


            old_pesq_score = pypesq(speech_y, mix_speech_y, sr)
            old_stoi_score = pystoi(speech_y, mix_speech_y, sr)
            old_snrseg_score = SNRseg(speech_y, mix_speech_y, fs=sr)

            new_dict["pesq"][snr]["val"] += (pesq_score - old_pesq_score)
            new_dict["pesq"][snr]["nums"] += 1

            new_dict["stoi"][snr]["val"] += ((stoi_score - old_stoi_score) / old_stoi_score) * 100
            new_dict["stoi"][snr]["nums"] += 1


            new_dict["ssnr"][snr]["val"] += (snrseg_score - old_snrseg_score)
            new_dict["ssnr"][snr]["nums"] += 1

            snrseg_list.append(snrseg_score)
            pesq_list.append(pesq_score)
            stoi_list.append(stoi_score)
            old_stoi_list.append(old_stoi_score)
            old_snrseg_list.append(old_snrseg_score)
            old_pesq_list.append(old_pesq_score)

    print(new_dict)


    print("pesq={:.4f} | stoi={:.4f} | seg_snr={:.4f} | old_pesq={:.4f} | old_stoi={:.4f} | old_seg_snr={:.4f}".format(
        sum(pesq_list) / len(pesq_list),
        sum(stoi_list) / len(stoi_list),
        sum(snrseg_list) / len(snrseg_list),
        sum(old_pesq_list) / len(old_pesq_list),
        sum(old_stoi_list) / len(old_stoi_list),
        sum(old_snrseg_list) / len(old_snrseg_list)))

    for score_class in new_dict:
        print("--------------------------------------------")
        print(score_class)
        for snr_val in new_dict[score_class]:
            print(f"snr:{snr_val} | mean_score:{new_dict[score_class][snr_val]['val'] / new_dict[score_class][snr_val]['nums']}")

if __name__ == "__main__":
    get_score()







