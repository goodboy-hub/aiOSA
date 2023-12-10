import numpy as np

'''
评价体系如下：
1.满足以下条件的直接评为最低分0
a.电压信号大于0.1的小于5个点,即电压信号以0.1为界,直接pass
b.第一个R峰过晚(第一个R峰在5秒之后),最后一个R峰过早(最后一个R峰在结束前5秒出现)
c.max(rri)> 3* mean(rri)
d.信号极差大于3mv(选用)

2.评分计算规则如下：
a.以R峰为中心取两秒的窗口,取很多窗口后叠加取均值得到一个平均窗口
b.将各个窗口与平均窗口正则化后计算Pearson相关系数
'''
class SignalQuality:

    def __init__(self):
        pass

    # def R_Wave_finetune(self, sig, rpos_lst):
    #     new_rpos_list = []
    #     inter = 20
    #     for rpos in rpos_lst:
    #         if rpos < 20 or rpos + 20 > len(sig):
    #             continue
    #         sub_sig = list(sig[rpos - inter:rpos + inter])
    #         new_rpos = rpos - inter + sub_sig.index(max(sub_sig))
    #         new_rpos_list.append(new_rpos)
    #     return new_rpos_list

    def preScreening(self, sig, rpos,fs=200):
        # rpos = self.R_Wave_finetune(sig, rpos)
        # Amplitude less than 3mV
        # ampl = np.abs(np.max(sig) - np.min(sig))
        #if ampl > 3:
        #    return False
        sig_len=len(sig)
        tmp_sig = np.abs(sig)
        tmp_sig = tmp_sig[tmp_sig > 0.1]
        if len(tmp_sig) < 5:
            return (False,'电压值过低')
        if rpos[0] > fs*5 or rpos[-1] < (sig_len-fs*5):
            return (False,'前方或后方有空缺')
        rr_intervals = np.diff(rpos)
        maxRR = np.max(rr_intervals)
        meanRR = np.mean(rr_intervals)
        if maxRR > meanRR * 3:
            return (False,'rri max值过大')
        return rpos

    def check_coeff(self, coeffs):
        coeffs = np.array(coeffs)
        count = 0
        while len(coeffs) > 0:
            coeffs = np.abs(coeffs - coeffs[0])
            coeffs = coeffs[coeffs > 0.1]
            count += 1
        return count

    def cal_corr_coeff_lst(self, sig, rpos):
        beat_seg = []
        for idx, r_p in enumerate(rpos):
            if r_p > 200 and (len(sig) - r_p) > 200:
                tmp_seg = sig[r_p - 200:r_p + 200]
                beat_seg.append(tmp_seg)
        beat_seg = np.array(beat_seg)
        template_qrs = np.mean(beat_seg, axis=0)
        template_qrs = template_qrs - np.mean(template_qrs)
        coeff_lst = []
        for seg in beat_seg:
            seg = seg - np.mean(seg)
            coeff = np.corrcoef(seg, template_qrs)[0, 1]
            coeff_lst.append(coeff)
        return coeff_lst

    def sqi(self, sig, rpos):
        preRes = self.preScreening(sig, rpos)

        if preRes[0]:
            rpos = rpos
        else:
            return (preRes[1], float(0))

        coeff_lst = self.cal_corr_coeff_lst(sig, rpos)
        # template_nums = self.check_coeff(coeff_lst)
        coeff = float(np.mean(coeff_lst))

        if coeff > 0.9:
            return ('pass', coeff)
        else:
            return ('pass', coeff)

        # if template_nums < 3:
        #     return True, coeff
        # else:
        #     return False, coeff
