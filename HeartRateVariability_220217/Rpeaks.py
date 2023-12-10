import numpy as np
import neurokit2 as nk
import scipy.signal as sps
import numpy as np
from scipy.interpolate import interp1d

def resample_interp(ts, fs_in, fs_out):
    """
    基于线性拟合的差值重采样算法
    计算前后点对应的比例进行插值
    :param ts:  单导联数据，一维浮点型数组
    :param fs_in: 原始采样率，整型
    :param fs_out: 目标采样率，整型
    :return: 重采样后的数据
    """
    t = len(ts) / fs_in
    fs_in, fs_out = int(fs_in), int(fs_out)
    if fs_out == fs_in:
        return np.array(ts)
    else:
        x_old = np.linspace(0, 1, num=len(ts), endpoint=True)
        x_new = np.linspace(0, 1, num=int(t * fs_out), endpoint=True)
        y_old = ts
        f = interp1d(x_old, y_old, kind='linear')
        y_new = f(x_new)
        return y_new

def simple_qrs_detector(ecg, fs):
    """
    简易的QRS波群检测器

    Parameters
    ----------
    ecg : 1D ECG数据
    fs : 信号采样率

    Returns
    -------
    QRS波群位置

    """
    fixed_fs = 500
    # remove power-line interference
    b, a = sps.iirnotch(50, 50, fs)
    ecg = sps.filtfilt(b, a, ecg)
    whole_fixed_fs_ecg = resample_interp(ecg, fs, fixed_fs)
    # simple filter
    b, a = sps.butter(N=4, Wn=[0.5, 45], btype='bandpass', fs=fixed_fs)
    whole_fixed_fs_ecg = sps.filtfilt(b, a, whole_fixed_fs_ecg)
    # clean ecg
    # clean ecg
    b, a = sps.butter(N=4, Wn=[3, 40], btype='bandpass', fs=fixed_fs)
    whole_fixed_fs_filtered = sps.filtfilt(b, a, whole_fixed_fs_ecg)
    # ecg r peaks
    _, info = nk.ecg_peaks(whole_fixed_fs_filtered, fixed_fs,
                           smoothwindow=0.2,
                           avgwindow=0.9,
                           gradthreshweight=1.2,
                           minlenweight=0.3,
                           mindelay=0.2)
    whole_fixed_fs_r_peaks = info['ECG_R_Peaks']
    if len(whole_fixed_fs_r_peaks) < 1:
        return whole_fixed_fs_r_peaks
    else:
        return [int(_ / fixed_fs * fs) for _ in whole_fixed_fs_r_peaks]

def R_Wave_finetune(sig, rpos_lst):
    new_rpos_list = []
    inter = 20
    for rpos in rpos_lst:
        if rpos < 20 or rpos + 20 > len(sig):
            continue
        sub_sig = list(sig[rpos - inter:rpos + inter])
        new_rpos = rpos - inter + sub_sig.index(max(sub_sig))
        new_rpos_list.append(new_rpos)
    return new_rpos_list
