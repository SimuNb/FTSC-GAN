import torch
from model.ftsc_gan import FTSC_GAN
import glob
import librosa
import tqdm
import numpy as np


def get_speech(epoch=0):
    batch_size = 8
    device = "cuda"
    torch.cuda.set_device(0)

    mix_speech_path = r"G:\paper\common_se_data\voicebank_train\test\mix"
    speech_path = r"G:\paper\common_se_data\voicebank_train\test\speech"

    mix_speech_list = glob.glob(mix_speech_path + "\\*")

    speech_list = glob.glob(speech_path + "\\*")

    generate_wav_path = r"G:\paper\common_se_data\voicebank_train\score_test_other\generate_wav\\"
    speech_wav_path = r"G:\paper\common_se_data\voicebank_train\score_test_other\speech_wav\\"
    mix_wav_path = r"G:\paper\common_se_data\voicebank_train\score_test_other\mix_wav\\"

    save_path = r"G:\paper\common_se_data\voicebank_train\sav_path\ftsc_gan_noisy_2.tar"
    generator = FTSC_GAN(conformer_nums=2).to(device)
    #
    if device == "cpu":
        checkpoint = torch.load(save_path, map_location="cpu")
    else:
        checkpoint = torch.load(save_path)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()

    print("epoch:", checkpoint["epoch"])

    for item in tqdm.tqdm(mix_speech_list):

        mix_y = np.load(item)

        final_list = []
        batch_times = int(mix_y.shape[0] / batch_size)
        this_speech_name = generate_wav_path + item.split("\\")[-1].replace("_mix.npy", ".npy")
        if batch_times == 0:
            one_batch = torch.Tensor(mix_y).unsqueeze(1).to(device)
            one_result = generator(one_batch).squeeze(1).detach().cpu().numpy()
            for j in range(len(one_result)):
                new_item = one_result[j]
                if j == 0:
                    final_list = new_item
                else:
                    final_list = np.hstack((final_list, new_item))

            np.save(this_speech_name, final_list)

        else:
            for i in range(batch_times):

                one_batch = mix_y[i * batch_size:(i + 1) * batch_size]

                one_batch = torch.Tensor(one_batch).float().unsqueeze(1).to(device)
                one_result = generator(one_batch).squeeze(1).detach().cpu().numpy()
                for j in range(len(one_result)):
                    new_item = one_result[j]
                    if i == 0 and j == 0:
                        final_list = new_item
                    else:
                        final_list = np.hstack((final_list, new_item))

                if i == batch_times - 1:
                    one_batch = mix_y[(i + 1) * batch_size:]
                    one_batch = torch.Tensor(one_batch).unsqueeze(1).to(device)
                    if one_batch.shape[0] == 0:
                        pass
                    else:
                        one_result = generator(one_batch).squeeze(1).detach().cpu().numpy()


                    for j in range(len(one_result)):
                        new_item = one_result[j]
                        final_list = np.hstack((final_list, new_item))
            np.save(this_speech_name, final_list)


    if epoch == 0:
        for item in tqdm.tqdm(speech_list):
            speech_y = np.load(item)
            one_list = []
            this_speech_name = speech_wav_path + item.split("\\")[-1].replace("_speech.npy", ".npy")
            for i in range(len(speech_y)):
                one = speech_y[i]
                if i == 0:
                    one_list = one
                else:
                    one_list = np.hstack((one_list, one))
            np.save(this_speech_name, one_list)

        for item in tqdm.tqdm(mix_speech_list):
            speech_y = np.load(item)
            mix_list = []
            this_speech_name = mix_wav_path + item.split("\\")[-1].replace("_mix.npy", ".npy")
            for i in range(len(speech_y)):
                one = speech_y[i]
                if i == 0:
                    mix_list = one
                else:
                    mix_list = np.hstack((mix_list, one))
            np.save(this_speech_name, mix_list)



get_speech()