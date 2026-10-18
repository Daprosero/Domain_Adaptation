# VENDORED UPSTREAM CODE -- copied verbatim from the paper's official repository.
# Paper: Zhu, Zhuang, Wang, Ke, Chen, Bian, Xiong, He. Deep Subdomain Adaptation Network for Image Classification. IEEE TNNLS 2021 (early access 2020). arXiv:2106.09388
# Repository: https://github.com/easezyc/deep-transfer-learning
# Commit: 2969ed6c740d04283aa54af4e2c267cb980c96fe
# Upstream path: UDA/pytorch1.0/DSAN/lmmd.py (whole file)
# License as found: MIT (LICENSE at repo root)
# Copied with the operator's explicit authorisation, cited here and in ../CITATIONS.md.
# Every deviation from the upstream bytes is marked with a `# LOCAL EDIT:` comment;
# all other lines (including their indentation) are unchanged.

import torch
import torch.nn as nn
import numpy as np

class LMMD_loss(nn.Module):
    def __init__(self, class_num=31, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super(LMMD_loss, self).__init__()
        self.class_num = class_num
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def get_loss(self, source, target, s_label, t_label):
        batch_size = source.size()[0]
        weight_ss, weight_tt, weight_st = self.cal_weight(
            s_label, t_label, batch_size=batch_size, class_num=self.class_num)
        # LOCAL EDIT: device-agnostic, no CUDA (upstream L37 was: weight_ss = torch.from_numpy(weight_ss).cuda())
        weight_ss = torch.from_numpy(weight_ss).to(source.device)
        # LOCAL EDIT: device-agnostic, no CUDA (upstream L38 was: weight_tt = torch.from_numpy(weight_tt).cuda())
        weight_tt = torch.from_numpy(weight_tt).to(source.device)
        # LOCAL EDIT: device-agnostic, no CUDA (upstream L39 was: weight_st = torch.from_numpy(weight_st).cuda())
        weight_st = torch.from_numpy(weight_st).to(source.device)

        kernels = self.guassian_kernel(source, target,
                                kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        # LOCAL EDIT: device-agnostic, no CUDA (upstream L43 was: loss = torch.Tensor([0]).cuda())
        loss = torch.Tensor([0]).to(source.device)
        if torch.sum(torch.isnan(sum(kernels))):
            return loss
        SS = kernels[:batch_size, :batch_size]
        TT = kernels[batch_size:, batch_size:]
        ST = kernels[:batch_size, batch_size:]

        loss += torch.sum(weight_ss * SS + weight_tt * TT - 2 * weight_st * ST)
        return loss

    def convert_to_onehot(self, sca_label, class_num=31):
        return np.eye(class_num)[sca_label]

    def cal_weight(self, s_label, t_label, batch_size=32, class_num=31):
        batch_size = s_label.size()[0]
        s_sca_label = s_label.cpu().data.numpy()
        s_vec_label = self.convert_to_onehot(s_sca_label, class_num=self.class_num)
        s_sum = np.sum(s_vec_label, axis=0).reshape(1, class_num)
        s_sum[s_sum == 0] = 100
        s_vec_label = s_vec_label / s_sum

        t_sca_label = t_label.cpu().data.max(1)[1].numpy()
        t_vec_label = t_label.cpu().data.numpy()
        t_sum = np.sum(t_vec_label, axis=0).reshape(1, class_num)
        t_sum[t_sum == 0] = 100
        t_vec_label = t_vec_label / t_sum

        index = list(set(s_sca_label) & set(t_sca_label))
        mask_arr = np.zeros((batch_size, class_num))
        mask_arr[:, index] = 1
        t_vec_label = t_vec_label * mask_arr
        s_vec_label = s_vec_label * mask_arr

        weight_ss = np.matmul(s_vec_label, s_vec_label.T)
        weight_tt = np.matmul(t_vec_label, t_vec_label.T)
        weight_st = np.matmul(s_vec_label, t_vec_label.T)

        length = len(index)
        if length != 0:
            weight_ss = weight_ss / length
            weight_tt = weight_tt / length
            weight_st = weight_st / length
        else:
            weight_ss = np.array([0])
            weight_tt = np.array([0])
            weight_st = np.array([0])
        return weight_ss.astype('float32'), weight_tt.astype('float32'), weight_st.astype('float32')
