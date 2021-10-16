import torch
import torchvision
import torch.optim
import networks
#import unet_oril as networks
import numpy as np
from PIL import Image
import glob
import time
import os
import torch.nn as nn

from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
# loader使用torchvision中自带的transforms函数
loader = transforms.Compose([
    transforms.ToTensor()])
unloader = transforms.ToPILImage()


def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image

def image_loader(image_name):
    image = Image.open(image_name).convert('RGB')
    image = loader(image).unsqueeze(0)
    return image.cuda()
img_type = "real"
def dehaze_image( image_depth_path,image_hazy_path,Id,spath,r):
    """
    当前，输入为有雾图像和无雾图像，有雾图像和无雾图像都转为灰度图进人网络
    :param image_down_path:深度图下采样，当前用有雾图像的灰度图代替
    :param image_label_path:深度图，用无雾图像灰度图代替
    :param image_add_path:有雾彩色图像
    :param Id:
    :return:
    """
    print(image_hazy_path,image_depth_path)
    img_hazy = Image.open(image_hazy_path).convert("RGB")
    img_hazy = img_hazy.resize((640, 480), Image.ANTIALIAS)
    img_depth = Image.open(image_depth_path)
    img_depth = img_depth.resize((640, 480), Image.ANTIALIAS)
    #data_hazy = data_hazy.convert("RGB")
    # data_hazy = data_hazy.resize((width, height),Image.ANTIALIAS)
    img_hazy = (np.asarray(img_hazy) / 255.0)

    img_depth = (np.asarray(img_depth) / 255.0)

    img_hazy = torch.from_numpy(img_hazy).float().permute(2, 0, 1).cuda().unsqueeze(0)
    #img_gt = torch.from_numpy(img_gt).float().permute(2, 0, 1).cuda().unsqueeze(0)
    #img_depth = torch.from_numpy(img_depth).float().permute(2, 0, 1).cuda().unsqueeze(0)
    img_depth = torch.from_numpy(img_depth).float().cuda().unsqueeze(0).unsqueeze(0)
    
    i = 5
    #dehaze_net = networks.IRN(i)



    #clean_image = dehaze_net(img_hazy, img_depth)#base
    clean_image,_ = dehaze_net(img_hazy, img_depth)
    #clean_image = tensor_to_PIL(clean_image).convert("RGB")
    # temp_tensor = clean_image[0].cuda().data.cpu().numpy()
    temp_tensor = (clean_image, 0)
    # clean_image = HDR.test(temp_tensor)

    # torchvision.utils.save_image(torch.cat((data_hazy, clean_image),0), "results_mish/" + image_path.split("/")[-1])
    # torchvision.utils.save_image((data_hazy,0), "results_mish/" + image_path.split("/")[-1])

    # clean_image = Image.fromarray(clean_image)
    temp = image_depth_path.split("/")[-1]
    # 	# clean_image.save("results_mish增强/"+temp.replace("\\",""))
    # cv2.imwrite("results_mish/" + temp.replace("合成","",temp),clean_image)
    # torchvision.utils.save_image(torch.cat((data_hazy, clean_image),0), "results_mish/" + image_path.split("/")[-1])
    if not os.path.exists(r'test_result/i%i_%s'%(r,img_type)):
        os.makedirs(r'test_result/i%i_%s/'%(r,img_type))
    # torchvision.utils.save_image(torch.cat((data_hazy, clean_image),0), "对比/i3-mish/" + temp.replace("\\",""))
    #clean_image.save('test_result/real/' + s + '/' + temp)
    torchvision.utils.save_image(clean_image, 'test_result/i%i_%s/'%(r,img_type)+ temp)


if __name__ == '__main__':
    #dehaze_net = networks.MSDFN()
    r = 5
    dehaze_net = networks.IRN(r)
    dehaze_net = nn.DataParallel(dehaze_net).cuda()
    s = ""
    dehaze_net.load_state_dict(torch.load('trained_model/i5_ITS_1/Epoch8.pth'))

    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    img_type = 'ITS_1'
    depth_list = glob.glob(r'/home/share/hd1/liwang/Dehaze Dataset/testdataset/SOTS-IN/depth_gray/*')
    hazy_list = glob.glob(r"/home/share/hd1/liwang/Dehaze Dataset/testdataset/SOTS-IN/hazy/*")
    #gt_list = glob.glob(r"/home/amax/share/FGD/IRDN-master/dataset/testdataset/outdoor/gt/*")
    s = time.time()
    for Id in range(len(depth_list)):
        dehaze_image(depth_list[Id],hazy_list[Id],Id,img_type,r)
        print(depth_list[Id], "done!")
    e = time.time()
    print((e-s)/492)