import torch,torchvision
import torch.optim
import os
import argparse
import time
import dataloader as dataloader
import networks
#import unet_oril as networks


from SSIM import SSIM
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def train(config):

    dehaze_net = networks.IRN(config.recurrent_iter)
    #dehaze_net = networks.MSDFN()
    dehaze_net = nn.DataParallel(dehaze_net).cuda()
    dehaze_net.apply(inplace_relu)
    # if config.epoched == None:
    #     pass
    # else:
    dehaze_net.load_state_dict(torch.load('trained_model/i5_ITS_1/Epoch8.pth'))
        
    # if config.in_or_out == "outdoor":
    #     train_dataset = dataloader.dehazing_loader(config.orig_images_path,
    #                                                config.hazy_images_path)
    # else:
    #     config.orig_images_path = "dataset/train_data/indoor/clear/"
    #     config.hazy_images_path = "dataset/train_data/indoor/hazy/"
    #     train_dataset = dataloader.dehazing_loader(config.orig_images_path,
    #                                                config.hazy_images_path)
    train_dataset = dataloader.dehazing_loader(config.orig_images_path, config.hazy_images_path,
                                               config.label_images_path)
    val_dataset = dataloader.dehazing_loader(config.orig_images_path_val, config.hazy_images_path_val,
                                             config.label_images_path_val, mode="val")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True,
                                               num_workers=config.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=False,
                                             num_workers=config.num_workers, pin_memory=True)

    if config.lossfunc == "MSE":
        criterion = nn.MSELoss().cuda()
    elif config.lossfunc == "SSIM":
        criterion = SSIM()
    else:													#MSE+SSIM Loss
        criterion1 = nn.MSELoss().cuda()
        criterion2 = SSIM()
    comput_ssim = SSIM() 



    dehaze_net.train()
    zt = 1
    Iters = 0
    indexX = []
    indexY = []
    for epoch in range(config.epoched,config.num_epochs):
        if epoch == 0:
            config.lr = 0.0003
        elif epoch >1 and epoch <=5:
            config.lr = 0.0001
        elif epoch >5 and epoch <=10:
            config.lr = 0.00006
        elif epoch >10 and epoch <=15:
            config.lr = 0.00003
        elif epoch >20 and epoch <=25:
            config.lr = 0.00001
        elif epoch >25 and epoch <=30:
            config.lr = 0.000006
        elif epoch >30 and epoch <=35:
            config.lr = 0.000003
        elif epoch >35:
            config.lr = 0.000001
        print("*" * 80 + "第%i轮" % epoch + "*" * 80)

        optimizer = torch.optim.AdamW(dehaze_net.parameters(), lr=config.lr)

        for iteration, (img_clean, img_haze,img_depth) in enumerate(train_loader):

            img_clean = img_clean.cuda()
            img_haze = img_haze.cuda()
            img_depth = img_depth.cuda()
            try:
                clean_image,_ = dehaze_net(img_haze,img_depth)
                if config.lossfunc == "MSE":
                    loss = criterion(clean_image,img_clean)
                elif config.lossfunc == "SSIM":

                    loss = criterion(img_clean,clean_image)
                    loss = -loss
                else:													
                    ssim = criterion2(img_clean,clean_image)
                    mse = criterion1(clean_image,img_clean)
                    loss = mse-ssim

                #del clean_image, img_clean
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm(dehaze_net.parameters(), config.grad_clip_norm)
                optimizer.step()
                Iters += 1
                if ((iteration + 1) % config.display_iter) == 0:
                    print("Loss at iteration", iteration + 1, ":", loss.item())

                if ((iteration + 1) % config.snapshot_iter) == 0:
                    torch.save(dehaze_net.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth')
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(e)
                    torch.cuda.empty_cache()
                else:
                    raise e
        _ssim=[]

        with torch.no_grad():

            for iteration, (clean, haze, depth) in enumerate(val_loader):
                clean = clean.cuda()
                haze = haze.cuda()
                depth = depth.cuda()

                clean_,__ = dehaze_net(haze,depth)
                _s = comput_ssim(clean,clean_)#计算ssim值
                _ssim.append(_s.item())

                torchvision.utils.save_image(torch.cat((haze,clean_,clean), 0),
                                             config.sample_output_folder + "/epoch%s"%epoch +"/"+ str(iteration + 1) + ".jpg")
                # try:
                #     torchvision.utils.save_image(_,config.sample_output_folder + "/epoch%s" % epoch + "/" + str(iteration + 1) + ".png")
                # except:
                #     print('save erour')
            _ssim = np.array(_ssim)

            print("-----The %i Epoch mean-ssim is :%f-----" %(epoch,np.mean(_ssim)))
            with open("trainlog/i5_ITS_2.log", "a+", encoding="utf-8") as f:
                s = "The %i Epoch mean-ssim is :%f" %(epoch,np.mean(_ssim))+ "\n"
                f.write(s)
            indexX.append(epoch+1)
            indexY.append(np.mean(_ssim))
        print(indexX,indexY)
        plt.plot(indexX,indexY,linewidth=2)
        plt.pause(0.1)
    plt.savefig("trainlog/i5_ITS_2.png")#i%i_%s-new
    torch.save(dehaze_net.state_dict(), config.snapshots_folder + "IRDN.pth")


if __name__ == "__main__":

  
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--orig_images_path', type=str,
                        default=r"/home/share/hd1/liwang/Dehaze Dataset/traindataset/ITS/gt/")
    parser.add_argument('--hazy_images_path', type=str,
                        default=r"/home/share/hd1/liwang/Dehaze Dataset/traindataset/ITS/hazy/")
    parser.add_argument('--label_images_path', type=str,
                        default=r"/home/share/hd1/liwang/Dehaze Dataset/traindataset/ITS/depth_gray/")

    parser.add_argument('--orig_images_path_val', type=str,
                        default=r"/home/share/hd1/liwang/Dehaze Dataset/testdataset/SOTS-IN/gt/")
    parser.add_argument('--hazy_images_path_val', type=str,
                        default=r"/home/share/hd1/liwang/Dehaze Dataset/testdataset/SOTS-IN/hazy/")
    parser.add_argument('--label_images_path_val', type=str,
                        default=r"/home/share/hd1/liwang/Dehaze Dataset/testdataset/SOTS-IN/depth_gray/")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--display_iter', type=int, default=1)
    parser.add_argument( '--snapshot_iter', type=int, default=10)
    parser.add_argument('--snapshots_folder', type=str, default="trained_model/i5_ITS_2/")
    parser.add_argument('--sample_output_folder', type=str, default="samples/i5_ITS_2/")
    parser.add_argument('--recurrent_iter', type=int, default=5)
    parser.add_argument('--in_or_out', type=str, default="indoor")
    parser.add_argument('--lossfunc', type=str, default="SSIM",help="choose Loss Function(MSE or -SSIM or MSE+SSIM).")
    parser.add_argument('--cudaid', type=str, default="0",help="choose cuda device id 0-7).")
    parser.add_argument('--epoched', type=int, default=0,help="choose cuda device id 0-7).")

    config = parser.parse_args()
    print(config)

    os.environ['CUDA_VISIBLE_DEVICES'] = config.cudaid

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)
    if not os.path.exists(config.sample_output_folder):
        os.mkdir(config.sample_output_folder)

    for i in range(config.num_epochs):
        path = config.sample_output_folder + "/epoch%s" % str(i)
        if not os.path.exists(path):
            os.mkdir(path)

    s = time.time()

    train(config)
    e = time.time()
    print(str(e-s))