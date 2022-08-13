import glob
import sys
import numpy as np
import tqdm

data_path = r"G:\paper_data\zhongwen\speech_noisy_5\dev\mix"
sav_path = r"G:\paper_data\zhongwen\speech_noisy_5\dev\new_mix"

mix_speech_list = glob.glob(data_path + "\*")

batch_size = 64

print(len(mix_speech_list))

i = 0
while i < (len(mix_speech_list) - 1):

    connect_time = 0
    mix_list = []
    speech_list = []
    noisy_list = []
    while connect_time < 400 and i < (len(mix_speech_list) - 1):
        sys.stdout.write("\rprocessing: %d / %d" % ((i + 1), len(mix_speech_list)))
        sys.stdout.flush()

        one_mix_speech = mix_speech_list[i]

        mix_speech_content = np.load(one_mix_speech)
        speech_speech_content = np.load(one_mix_speech.replace("mix", "speech"))
        noisy_speech_content = np.load(one_mix_speech.replace("mix", "noisy"))


        for j in range(len(mix_speech_content)):
            one_mix = mix_speech_content[j]
            one_speech = speech_speech_content[j]
            one_noisy = noisy_speech_content[j]
            mix_list.append(one_mix)
            speech_list.append(one_speech)
            noisy_list.append(one_noisy)

        connect_time += 1
        i += 1

    assert len(mix_list) == len(speech_list) == len(noisy_list)
    state = np.random.get_state()
    np.random.shuffle(mix_list)

    np.random.set_state(state)
    np.random.shuffle(speech_list)

    np.random.set_state(state)
    np.random.shuffle(noisy_list)

    len_num = int(len(mix_list) / batch_size)

    for j in tqdm.tqdm(range(0, len_num * batch_size, batch_size)):
        new_mix_file = mix_list[j:j + batch_size]
        new_speech_file = speech_list[j:j + batch_size]
        new_noisy_file = noisy_list[j:j + batch_size]

        if i / 400 > int(i / 400):
            num = int(i / 400) + 1
        else:
            num = int(i / 400)

        mix_name = str(num) + "_" + str(int(j / batch_size)) + "_mix.npy"
        mix_path = sav_path + "\\" + mix_name

        speech_name = str(num) + "_" + str(int(j / batch_size)) + "_speech.npy"
        speech_path = sav_path.replace("new_mix", "new_speech") + "\\" + speech_name

        noisy_name = str(num) + "_" + str(int(j / batch_size)) + "_noisy.npy"
        noisy_path = sav_path.replace("new_mix", "new_noisy") + "\\" + noisy_name


        np.save(mix_path, new_mix_file)
        np.save(speech_path, new_speech_file)
        np.save(noisy_path, new_noisy_file)























