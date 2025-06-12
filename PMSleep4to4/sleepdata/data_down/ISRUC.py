import requests

import os

def mat_download1():
    url ="http://dataset.isr.uc.pt/ISRUC_Sleep/ExtractedChannels/subgroupI-Extractedchannels/subject1.mat"
    save_path = r"E:\sleepstage\sleepdata\ISRUC\firstsub\1"
    for index in range(20, 101):
        url_index = str(index) + ".mat"
        url_new = url.replace("1.mat", url_index)
        save_path_index = save_path.replace("1", str(index))
        save_path_new = os.path.join(save_path_index, url_index)
        #下载文件
        r = requests.get(url_new, allow_redirects=True)
        #保存文件
        open(save_path_new, 'wb').write(r.content)
        print("下载完成：", url_index)
def mat_download2():
    url1 ="http://dataset.isr.uc.pt/ISRUC_Sleep/ExtractedChannels/subgroupII-Extractedchannels/1/subject1.mat"
    save_path = r"E:\sleepstage\sleepdata\ISRUC\secondsub\1"
    url2 ="http://dataset.isr.uc.pt/ISRUC_Sleep/ExtractedChannels/subgroupII-Extractedchannels/2/subject1.mat"
    for index in range(1, 9):
        url_index = str(index) + ".mat"
        url1_new = url1.replace("1.mat", url_index)
        url2_new = url2.replace("1.mat", url_index)
        save_path_index = save_path.replace("1", str(index))
        save_path1 = save_path_index + r"\1"
        save_path1 = os.path.join(save_path1, url_index)
        save_path2 = save_path_index + r"\2"
        save_path2 = os.path.join(save_path2, url_index)
        #下载文件
        r = requests.get(url1_new, allow_redirects=True)

        #保存文件
        open(save_path1, 'wb').write(r.content)
        print("下载完成：", index,"1")
        #下载文件
        r = requests.get(url2_new, allow_redirects=True)
        #保存文件
        open(save_path2, 'wb').write(r.content)
        print("下载完成：", index,"2")
def mat_download3():
    url ="http://dataset.isr.uc.pt/ISRUC_Sleep/ExtractedChannels/subgroupIII-Extractedchannels/subject1.mat"
    save_path = r"E:\sleepstage\sleepdata\ISRUC\thirdsub\1"
    for index in range(1, 101):
        url_index = str(index) + ".mat"
        url_new = url.replace("1.mat", url_index)
        save_path_index = save_path.replace("1", str(index))
        save_path_new = os.path.join(save_path_index, url_index)
        #下载文件
        r = requests.get(url_new, allow_redirects=True)
        #保存文件
        open(save_path_new, 'wb').write(r.content)
        print("下载完成：", url_index)
if __name__ == '__main__':
    mat_download1()
    mat_download2()
    mat_download3()