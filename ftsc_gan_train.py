import time
import torch
import warnings
import torch.nn as nn
import numpy as np
import glob
import pysepm
from utils.DateLoader import data_loader, no_shuffle_data_loader
from utils.utils import transform_pesq_range
from model.ftsc_gan import FTSC_GAN
from model.discriminator_t import Discriminator_T

warnings.filterwarnings("ignore")  # 忽略告警

epochs = 70
bad_times = 30
info_fre = 1000
batch_size = 8
draw_fre = 5
betas = (0.5, 0.999)
sr = 16000
save_name = f"ftsc_gan_noisy_2.tar"
print(save_name)

torch.cuda.set_device(0)
device = "cuda"

train_data_path = r"G:\paper\common_se_data\voicebank_train\train\mix\\"

sav_path = r"G:\paper\common_se_data\voicebank_train\sav_path\\"

# load training-data

generator = FTSC_GAN(conformer_nums=2).to(device)
discriminator = Discriminator_T().to(device)
gan_criterion = nn.MSELoss()
l1_criterion = nn.L1Loss()

generator_optimizier = torch.optim.Adam(generator.parameters(), betas=betas, lr=1e-3)
discriminator_optimizier = torch.optim.RMSprop(discriminator.parameters(), lr=1e-4)

g_scheduler = torch.optim.lr_scheduler.StepLR(generator_optimizier, step_size=1, gamma=0.88)
d_scheduler = torch.optim.lr_scheduler.StepLR(discriminator_optimizier, step_size=1, gamma=0.9)


def get_speech(epoch, generator):
    device = "cuda"
    torch.cuda.set_device(0)

    mix_speech_path = r"G:\paper\common_se_data\voicebank_train\test\mix"
    speech_path = r"G:\paper\common_se_data\voicebank_train\test\speech"

    mix_speech_list = glob.glob(mix_speech_path + "\\*")

    speech_list = glob.glob(speech_path + "\\*")

    generate_wav_path = r"G:\paper\common_se_data\voicebank_train\socre_test\generate_wav\\"
    speech_wav_path = r"G:\paper\common_se_data\voicebank_train\socre_test\speech_wav\\"
    mix_wav_path = r"G:\paper\common_se_data\voicebank_train\socre_test\mix_wav\\"

    print("epoch:", epoch)
    time.sleep(0.1)

    for item in mix_speech_list:

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

    if epoch == 1:
        for item in speech_list:
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

        for item in mix_speech_list:
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

def get_score():
    sr = 16000
    speech_wav = r"G:\paper\common_se_data\voicebank_train\socre_test\speech_wav\\"

    speech_wav_list = glob.glob(speech_wav + "*")
    snrseg_list = []
    pesq_list = []
    stoi_list = []
    csig_list = []
    cbak_list = []
    covl_list = []


    for i in range(len(speech_wav_list)):
        item = speech_wav_list[i]
        gen_speech = item.replace("speech_wav", "generate_wav")
        l = int(item.split("\\")[-1].replace(".npy", "").split("_")[-1])

        speech_y = np.load(item)[:l]
        gen_speech_y = np.load(gen_speech)[:l]

        if len(speech_y) == len(gen_speech_y):
            try:
                pesq_score = pysepm.pesq(speech_y, gen_speech_y, sr)[1]
                stoi_score = pysepm.stoi(speech_y, gen_speech_y, sr)
                snrseg_score = pysepm.SNRseg(speech_y, gen_speech_y, sr)
                csig, cbak, covl = pysepm.composite(speech_y, gen_speech_y, sr)

                snrseg_list.append(snrseg_score)
                pesq_list.append(pesq_score)
                stoi_list.append(stoi_score)
                csig_list.append(csig)
                cbak_list.append(cbak)
                covl_list.append(covl)

                pesq_list.append(pesq_score)
                stoi_list.append(stoi_score)
            except:
                print(item)

    print("pesq={:.4f} | stoi={:.4f} | seg_snr={:.4f} | CSIG={:.4f} | CBAK={:.4f} | COVL={:.4f}".format(
        sum(pesq_list) / len(pesq_list),
        sum(stoi_list) / len(stoi_list),
        sum(snrseg_list) / len(snrseg_list),
        sum(csig_list) / len(csig_list),
        sum(cbak_list) / len(cbak_list),
        sum(covl_list) / len(covl_list),
    ))
    return sum(pesq_list) / len(pesq_list),\
        sum(stoi_list) / len(stoi_list),\
        sum(snrseg_list) / len(snrseg_list),\
        sum(csig_list) / len(csig_list),\
        sum(cbak_list) / len(cbak_list),\
        sum(covl_list) / len(covl_list)\



def train_one_step(clean_speechs, mixed_speechs, is_train):
    clean_speechs = clean_speechs.unsqueeze(2).permute(0, 2, 1)
    mixed_speechs = mixed_speechs.unsqueeze(2).permute(0, 2, 1)

    # -----------------------------train discriminator------------------------------

    discriminator_optimizier.zero_grad()

    generator_fake_speech = generator(mixed_speechs).detach()
    noisy_fake_pred = discriminator(torch.cat([mixed_speechs - generator_fake_speech, mixed_speechs], dim=1))
    noisy_real_pred = discriminator(torch.cat([mixed_speechs - clean_speechs, mixed_speechs], dim=1))

    noisy_fake_loss = gan_criterion(noisy_fake_pred, torch.zeros_like(noisy_fake_pred))
    noisy_real_loss = gan_criterion(noisy_real_pred, torch.ones_like(noisy_real_pred))

    discriminator_loss = 0.5 * noisy_fake_loss + 0.5 * noisy_real_loss

    discriminator_loss.backward()
    discriminator_optimizier.step()

    # -----------------------------train generator--------------------------------
    generator_optimizier.zero_grad()

    generator_fake_speech = generator(mixed_speechs)

    noisy_fake_pred = discriminator(torch.cat([mixed_speechs - generator_fake_speech, mixed_speechs], dim=1))
    gan_loss = gan_criterion(noisy_fake_pred, torch.ones_like(noisy_fake_pred))

    l1_loss = l1_criterion(mixed_speechs - generator_fake_speech, mixed_speechs - clean_speechs)

    generator_loss = gan_loss + 100 * l1_loss

    generator_loss.backward()
    generator_optimizier.step()
    return generator_loss, discriminator_loss, gan_loss


def load_model_from_checkpoint(save_path):
    global generator, generator_optimizier
    checkpoint = torch.load(save_path)
    generator.load_state_dict(checkpoint["generator"])
    generator_optimizier.load_state_dict(checkpoint["model_optimizier"])
    return checkpoint["epoch"]


def save_model_to_checkpoint(save_path, epoch, generator, generator_optimizier, discriminator, discriminator_optimizier):
    torch.save({
        "epoch": epoch + 1,
        "generator": generator.state_dict(),
        "generator_optimizier": generator_optimizier.state_dict(),
        "discriminator": discriminator.state_dict(),
        "discriminator_optimizier": discriminator_optimizier.state_dict(),

    }, save_path)


def train(epochs):
    times = []
    start = time.time()

    count = 0
    best_score = 0

    d_loss_list = []
    g_loss_list = []
    gan_loss_list = []
    for epoch in range(epochs):
        fw = open("records/all_snr.txt", "a", encoding="utf-8")

        train_data = data_loader(train_data_path, batch_size, 400)
        # train_data = no_shuffle_data_loader(train_data_path)

        if epoch == 0:
            print("-" * 100)
            print("train......")
            print("-" * 100)
        if count < bad_times:
            generator.train()
            for i, data in enumerate(train_data):
                mix_data, clean_data = data

                mix_train_data = torch.Tensor(mix_data)
                clean_trian_data = torch.Tensor(clean_data)

                clean_trian_data = clean_trian_data.to(device)
                mix_train_data = mix_train_data.to(device)
                generator_loss, discriminator_loss, gan_loss = train_one_step(clean_trian_data, mix_train_data, is_train=True)
                g_loss_list.append(generator_loss.item())
                d_loss_list.append(discriminator_loss.item())
                gan_loss_list.append(gan_loss.item())

                if (i + 1) % info_fre == 0:
                    print("[{}/{}]:stpes={} | generator_loss={:.4f} | discriminator_loss={:.4f} | gan_loss={:.4f}".format(epoch + 1,
                                                                                                                          epochs,
                                                                                                                          i + 1,
                                                                                                                          generator_loss.item(),
                                                                                                                          discriminator_loss.item(),
                                                                                                                          gan_loss.item())),

            print("-" * 100)
            print("[{}/{}]: avg_g_loss={:.4f} | avg_d_loss={:.4f} | avg_gan_loss={:.4f}".format(epoch + 1,
                                                                          epochs,
                                                                          sum(g_loss_list) / len(g_loss_list),
                                                                          sum(d_loss_list) / len(d_loss_list),
                                                                          sum(gan_loss_list) / len(gan_loss_list))),
            print("-" * 100)

            # if epoch % draw_fre == 0:
            #     plt.plot(gan_loss_list, color="red")
            #     plt.plot(d_loss_list, color="green")
            #     plt.show()

            print("test......")
            generator.eval()
            get_speech(epoch + 1, generator)
            pesq_s, stoi_s, ssnr, cisg, cbak, covl = get_score()
            score = (stoi_s + transform_pesq_range(pesq_s) + (cisg + cbak + covl - 3)/4) / 2
            # fw.write(f"epoch:{epoch + 1} | pesq:{pesq_s}" + "\n")
            # fw.close()

            g_scheduler.step()
            d_scheduler.step()



            if best_score == 0:
                best_score = score
                save_model_to_checkpoint(sav_path + save_name, epoch, generator, generator_optimizier, discriminator, discriminator_optimizier)
                print("save model to path of save_model")
                print("-" * 100)
            else:
                if score > best_score:
                    best_score = score
                    save_model_to_checkpoint(sav_path + save_name, epoch, generator, generator_optimizier, discriminator, discriminator_optimizier)
                    count = 0
                    if count % 5:
                        g_scheduler.step()
                        d_scheduler.step()
                    print("save model to path of save_model")
                    print("-" * 100)
                else:
                    count += 1
                    print(f"do not save model {count} time!")
                    print("-" * 100)

        else:
            print("best epoch!")
            break

    print('== Finish Training ==')
    print('Time for epochs {} is {:.4f} sec'.format(epochs, time.time() - start))
    times.append(time.time() - start),

def main():
    train(epochs)


if __name__ == "__main__":
    main()