import torch
import torch.nn as nn

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class CIFPN(nn.Module):
    def __init__(self, num_features):
        super(CIFPN, self).__init__()
        # 每层与其他层之间的动态权重
        self.weights = nn.ModuleList([nn.ModuleList(
            [nn.ParameterList([nn.Parameter(torch.ones(1, dtype=torch.float32)) for _ in range(num_features)]) for _ in
             range(num_features)]
        ) for _ in range(num_features)])
        self.swish = Swish()
        self.epsilon = 0.0001

    def forward(self, x):
        # 确保输入特征为4维
        x = [xi.squeeze(0) if xi.dim() == 5 else xi for xi in x]
        x = [xi if xi.dim() == 4 else xi.unsqueeze(0) for xi in x]  # 确保输入为4维

        # ========== 底-上路径 ==========
        bottom_up_outputs = []
        for i in range(len(x)):  # 当前层 i
            weighted_inputs = []
            for j in range(len(x)):  # 遍历所有层 j，包括同级层
                weights = self.weights[i][j]  # 动态权重
                normalized_weights = [weights[k] / (torch.sum(self.swish(weights[k])) + self.epsilon) for k in
                                      range(len(weights))]

                # 对 j 层的特征进行插值调整至与 i 层特征尺寸一致
                resized_features = [torch.nn.functional.interpolate(x[k], size=x[i].shape[-2:], mode='nearest') for
                                    k in range(len(x))]

                # 加权融合所有层的特征，包括同级层
                weighted_sum = sum([normalized_weights[k] * resized_features[k] for k in range(len(x))])
                weighted_inputs.append(weighted_sum)

            # 当前层的最终输出融合所有输入特征
            fused_feature_map = sum(weighted_inputs) if weighted_inputs else x[i]
            bottom_up_outputs.append(fused_feature_map)

        # ========== 上-下路径 ==========
        top_down_outputs = [bottom_up_outputs[-1]]  # 从最高层开始
        for i in range(len(bottom_up_outputs) - 2, -1, -1):  # 从倒数第二层到最底层
            weighted_inputs = [top_down_outputs[-1]]  # 包括从上一层传递下来的特征
            for j in range(len(bottom_up_outputs)):  # 遍历所有层 j，包括同级层
                weights = self.weights[i][j]
                normalized_weights = [weights[k] / (torch.sum(self.swish(weights[k])) + self.epsilon) for k in
                                      range(len(weights))]

                # 对 j 层的特征进行插值调整至与 i 层特征尺寸一致
                resized_features = [
                    torch.nn.functional.interpolate(bottom_up_outputs[k], size=bottom_up_outputs[i].shape[-2:],
                                                    mode='nearest') for k in range(len(bottom_up_outputs))]

                # 加权融合所有层的特征，包括同级层
                weighted_sum = sum(
                    [normalized_weights[k] * resized_features[k] for k in range(len(bottom_up_outputs))])
                weighted_inputs.append(weighted_sum)

            # 当前层的最终输出融合所有输入特征
            fused_feature_map = sum(weighted_inputs)
            top_down_outputs.append(fused_feature_map)

        # 翻转上-下路径输出以匹配原始层顺序
        top_down_outputs = top_down_outputs[::-1]

        # ========== 底-上和上-下路径的最终融合 ==========
        fused_outputs = [top_down_outputs[i] + bottom_up_outputs[i] for i in range(len(x))]
        return torch.cat(fused_outputs, dim=1)[:, :64, :, :]  # 最终输出沿通道维度拼接，并裁剪到64通道