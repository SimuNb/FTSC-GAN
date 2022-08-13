import glob
import numpy as np

def no_shuffle_data_loader(mix_path):
    mix_speech_list = glob.glob(mix_path + "*")
    np.random.shuffle(mix_speech_list)
    i = 0
    while i < len(mix_speech_list) - 1:
        one_mix_speech = mix_speech_list[i]
        mix_speech_content = np.load(one_mix_speech)
        speech_speech_content = np.load(one_mix_speech.replace("mix", "speech"))
        i += 1
        yield (mix_speech_content, speech_speech_content)




def data_loader(mix_path, batch_size, file_nums):

    mix_speech_list = glob.glob(mix_path + "*")
    np.random.shuffle(mix_speech_list)

    i = 0
    mix_center_list = []
    speech_center_list = []
    while i < (len(mix_speech_list) - 1):

        connect_time = 0
        mix_list = mix_center_list
        speech_list = speech_center_list
        # noisy_list = []
        while connect_time < file_nums and i < (len(mix_speech_list) - 1):
            one_mix_speech = mix_speech_list[i]

            mix_speech_content = np.load(one_mix_speech)
            speech_speech_content = np.load(one_mix_speech.replace("mix", "speech"))
            # noisy_speech_content = np.load(one_mix_speech.replace("mix", "noisy"))

            for j in range(len(mix_speech_content)):
                one_mix = mix_speech_content[j]
                one_speech = speech_speech_content[j]
                # one_noisy = noisy_speech_content[j]

                mix_list.append(one_mix)
                speech_list.append(one_speech)
                # noisy_list.append(one_noisy)

            connect_time += 1
            i += 1

        assert len(mix_list) == len(speech_list)
        state = np.random.get_state()
        np.random.shuffle(mix_list)

        np.random.set_state(state)
        np.random.shuffle(speech_list)


        len_num = int(len(mix_list) / batch_size)

        mix_center_list = mix_list[len_num * batch_size:]
        speech_center_list = speech_list[len_num * batch_size:]

        for j in range(0, len_num * batch_size, batch_size):
            new_mix_file = mix_list[j:j + batch_size]
            new_speech_file = speech_list[j:j + batch_size]

            yield (new_mix_file, new_speech_file)