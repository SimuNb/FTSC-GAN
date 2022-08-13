import pysepm
import torch
import numpy as np

def transform_pesq_range(pesq_score):
    return (pesq_score + 0.5) / 5

def softmax(scores):
    softmax = np.exp(scores) / np.sum(np.exp(scores))
    return softmax

def frame_score(real_speech,fake_speech, sr):
    score_list = []
    assert len(fake_speech) == len(real_speech)

    for i in range(len(real_speech)):
        try:
            one_fake = fake_speech[i].cpu().numpy()
            one_real = real_speech[i].cpu().numpy()

            stoi = pysepm.stoi(one_real, one_fake, sr)
            pesq, _ = pysepm.pesq(one_real, one_fake, sr)
            normal_pesq = transform_pesq_range(pesq)

            score = (stoi + normal_pesq) / 2
            score_list.append(score)

        except:
            with open("error.txt", "a", encoding="utf-8") as fw:
                fw.write(f"stoi:{stoi + 0.2}")
                fw.write("\n")
                fw.close()
            if stoi > 0.15:
                score_list.append(stoi)
            elif stoi > 0:
                score_list.append(stoi + 0.1)
            else:
                score_list.append(stoi + 0.2)
    return torch.Tensor(score_list)

def no_shuffle_frame_score(real_tensor, fake_tensor, sr):
    real_speech = real_tensor.view(-1).cpu().numpy()
    fake_speech = fake_tensor.view(-1).cpu().numpy()
    assert len(fake_speech) == len(real_speech)

    stoi = pysepm.stoi(real_speech, fake_speech, sr)
    pesq, _ = pysepm.pesq(real_speech, fake_speech, sr)
    normal_pesq = transform_pesq_range(pesq)
    score = (stoi + normal_pesq) / 2
    score_list = [score] * len(real_tensor)

    return torch.Tensor(score_list)

def padding_zero_num(num_list):
    num = 0
    for i in range(len(num_list)):
        if num_list[len(num_list) - 1 - i] == 0:
            num += 1
        else:
            break
    return num

if __name__ == '__main__':
    import glob
    import tqdm
    fake_path = r"G:\paper_data\zhongwen\speech_noisy_0\test\mix"
    real_path = r"G:\paper_data\zhongwen\speech_noisy_0\test\speech"
    fake_list = glob.glob(fake_path + "\\*")
    real_list = glob.glob(real_path + "\\*")
    for item in tqdm.tqdm(fake_list):
        try:
            fake = torch.Tensor(np.load(item))
            real = torch.Tensor(np.load(item.replace("mix", "speech")))
            a = frame_score(fake, real, 8000)
        except:
            print(item)
