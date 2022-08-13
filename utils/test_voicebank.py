import glob
import tqdm
import numpy as np
import librosa
import warnings
from pypesq import pesq as pypesq
from pystoi import stoi as pystoi
import pysepm

warnings.filterwarnings("ignore")  # 忽略告警

new_dict = {"pesq": {-2.5: {"val": 0, "nums": 0}, 2.5: {"val": 0, "nums": 0}, 7.5: {"val": 0, "nums": 0},
                     12.5: {"val": 0, "nums": 0}},
            "stoi": {-2.5: {"val": 0, "nums": 0}, 2.5: {"val": 0, "nums": 0}, 7.5: {"val": 0, "nums": 0},
                     12.5: {"val": 0, "nums": 0}},
            "ssnr": {-2.5: {"val": 0, "nums": 0}, 2.5: {"val": 0, "nums": 0}, 7.5: {"val": 0, "nums": 0},
                     12.5: {"val": 0, "nums": 0}}}


def get_score():
    speech_wav = r"G:\paper\common_se_data\voicebank_train\score_test_other\speech_wav"
    speech_wav_list = glob.glob(speech_wav + "\\*")

    snrseg_list = []
    pesq_list = []
    stoi_list = []
    csig_list = []
    cbak_list = []
    covl_list = []

    sr = 16000

    for i in tqdm.tqdm(range(len(speech_wav_list))):
        clean_speech = speech_wav_list[i]
        noisy = clean_speech.replace("speech_wav", "generate_wav")
        l = int(noisy.split("\\")[-1].replace(".npy", "").split("_")[-1])

        mix_y = np.load(noisy)
        speech_y = np.load(clean_speech)
        mix_y = mix_y[:l]
        speech_y = speech_y[:l]

        if len(speech_y) == len(mix_y):

            pesq_score = pysepm.pesq(speech_y, mix_y, sr)[1]
            stoi_score = pysepm.stoi(speech_y, mix_y, sr)
            snrseg_score = pysepm.SNRseg(speech_y, mix_y, sr)
            csig, cbak, covl = pysepm.composite(speech_y, mix_y, sr)

            snrseg_list.append(snrseg_score)
            pesq_list.append(pesq_score)
            stoi_list.append(stoi_score)
            csig_list.append(csig)
            cbak_list.append(cbak)
            covl_list.append(covl)


    print(new_dict)

    print("pesq={:.4f} | stoi={:.4f} | seg_snr={:.4f} | CSIG={:.4f} | CBAK={:.4f} | COVL={:.4f}".format(
        sum(pesq_list) / len(pesq_list),
        sum(stoi_list) / len(stoi_list),
        sum(snrseg_list) / len(snrseg_list),
        sum(csig_list) / len(csig_list),
        sum(cbak_list) / len(cbak_list),
        sum(covl_list) / len(covl_list),
    ))

    # for score_class in new_dict:
    #     print("--------------------------------------------")
    #     print(score_class)
    #     for snr_val in new_dict[score_class]:
    #         print(
    #             f"snr:{snr_val} | mean_score:{new_dict[score_class][snr_val]['val'] / new_dict[score_class][snr_val]['nums']}")
    #

if __name__ == "__main__":
    get_score()







