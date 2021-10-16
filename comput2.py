import cv2
#from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity as compare_ssim
#from skimage import measure
from skimage.metrics import peak_signal_noise_ratio
import numpy as np
import glob
import os
from PIL import Image
def comput(path1,path2):


    print(path1,path2)
    img1 = Image.open(path1)
    img2 = Image.open(path2)
    #print(img2.size)
    img1 = img1.resize((640, 480), Image.ANTIALIAS)
    img2 = img2.resize((640, 480), Image.ANTIALIAS)
    #img2 = img2.resize(img1.size, Image.ANTIALIAS)
    img1 = np.asarray(img1)
    img2 = np.asarray(img2)


    # print(path1,path2)
    # img1 = cv2.imread(path1)
    # img2 = cv2.imread(path2)
    # img1 = cv2.resize(img1, (640, 480))
    # img2 = cv2.resize(img2, (640, 480))
    SSIM = compare_ssim(img1, img2,multichannel=True)
    PSNR = peak_signal_noise_ratio(img1, img2)
    return SSIM,PSNR
def preprocess(path1,path2):
    imagename = path1.split("/")[-1].split('.jpg')[0]
    print('name is :', imagename)
    imagetype = ".jpg"
    path = path2 + imagename #+ imagetype
    #path = path2+imagename.split("_")[0]+"_"+imagename.split("_")[1]+"_GT"+imagetype#I-HAZE O-HAZE
    return path
if __name__ == "__main__":
    #mathod_list = ['DCP','Haze line','MSCNN','AOD-Net','GCANet','GridDehazeNet','FFA-Net','MSBDN','OURS','IRN','RefinedNet','PPFNet','MSDFN']
    mathod_list = ['OURS']
    mathod_name = "OURS"
    dataset_name = "ITS"
    gt_List = glob.glob(r"/home/share/hd1/liwang/Dehaze Dataset/testdata/dataset/%s/gt/*" % dataset_name)
    respath = r"/home/share/hd1/liwang/Dehaze Dataset/testdata/%s/%s/" % (dataset_name,mathod_name)
    #_list = glob.glob(r"/home/amax/share/FGD/DEHAZE-METHOD/GridDehazeNet/outdoor_results")
    SSIM = []
    PSNR = []

    for i in range(len(gt_List)):
        path = preprocess(gt_List[i],respath)
        path = path.replace('GT','hazy')
        try:
            ssim,psnr = comput(gt_List[i],path)
            SSIM.append(ssim)
            PSNR.append(psnr)
        except Exception as E:
            print(path,"文件未找到",E)
    n = np.array([SSIM,PSNR])
    num = mathod_list.index(mathod_name)+1
    np.save('test data/%s/%i_%s_%s.npy'%(dataset_name,num,mathod_name,dataset_name),n)
    print(len(SSIM))
    s = np.array(SSIM)
    p = np.array(PSNR)
    print(np.mean(s),np.mean(p))