import torch
import torch.nn as nn
import torch.nn.functional as F

from weaver.nn.model.ParticleTransformer import ParticleTransformer
from weaver.utils.logger import _logger

'''
Link to the full model implementation:
https://github.com/hqucms/weaver-core/blob/main/weaver/nn/model/ParticleTransformer.py
'''

class ParticleFlowNetwork(nn.Module):
    r"""Parameters
    ----------
    input_dims : int
        Input feature dimensions.
    num_classes : int
        Number of output classes.
    layer_params : list
        List of the feature size for each layer.
    """

    def __init__(self, input_dims, num_classes,
                 Phi_sizes=(100, 100, 128),
                 F_sizes=(100, 100, 100),
                 use_bn=True,
                 for_inference=False,
                 **kwargs):

        super(ParticleFlowNetwork, self).__init__(**kwargs)
        # input bn
        self.input_bn = nn.BatchNorm1d(input_dims) if use_bn else nn.Identity()
        # per-particle functions
        phi_layers = []
        for i in range(len(Phi_sizes)):
            phi_layers.append(nn.Sequential(
                nn.Conv1d(input_dims if i == 0 else Phi_sizes[i - 1], Phi_sizes[i], kernel_size=1),
                nn.BatchNorm1d(Phi_sizes[i]) if use_bn else nn.Identity(),
                nn.ReLU())
            )
        self.phi = nn.Sequential(*phi_layers)
        # global functions
        f_layers = []
        for i in range(len(F_sizes)):
            f_layers.append(nn.Sequential(
                nn.Linear(Phi_sizes[-1] if i == 0 else F_sizes[i - 1], F_sizes[i]),
                nn.ReLU())
            )
        f_layers.append(nn.Linear(F_sizes[-1], num_classes))
        if for_inference:
            f_layers.append(nn.Softmax(dim=1))
        self.fc = nn.Sequential(*f_layers)

    def forward(self, points, features, lorentz_vectors, mask):
        # x: the feature vector initally read from the data structure, in dimension (N, C, P)
        x = self.input_bn(features)
        x = self.phi(x)
        if mask is not None:
            x = x * mask.bool().float()
        x = x.sum(-1)
        return self.fc(x)


class ParticleTransformerWrapper(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        in_dim = kwargs['embed_dims'][-1]
        fc_params = kwargs.pop('fc_params')
        num_classes = kwargs.pop('num_classes')
        self.for_inference = kwargs['for_inference']

        fcs = []
        for out_dim, drop_rate in fc_params:
            fcs.append(nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(drop_rate)))
            in_dim = out_dim
        fcs.append(nn.Linear(in_dim, num_classes))
        self.fc = nn.Sequential(*fcs)

        kwargs['num_classes'] = None
        kwargs['fc_params'] = None
        self.mod = ParticleTransformer(**kwargs)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mod.cls_token', }

    def forward(self, points, features, lorentz_vectors, mask):
        x_cls = self.mod(features, v=lorentz_vectors, mask=mask)
        output = self.fc(x_cls)
        if self.for_inference:
            output = torch.softmax(output, dim=1)
        return output
    # below is normal ParT
    # def __init__(self, **kwargs) -> None:
    #     super().__init__()
    #     self.mod = ParticleTransformer(**kwargs)

    # @torch.jit.ignore
    # def no_weight_decay(self):
    #     return {'mod.cls_token', }

    # def forward(self, points, features, lorentz_vectors, mask):
    #     return self.mod(features, v=lorentz_vectors, mask=mask)



def get_teacher_model(data_config, **kwargs):

    cfg = dict(
            input_dim=len(data_config.input_dicts['pf_features']),
            num_classes=len(data_config.label_value),
            # network configurations
            pair_input_dim=4,
            use_pre_activation_pair=False,
            embed_dims=[128, 512, 128],
            pair_embed_dims=[64, 64, 64],
            num_heads=8,
            num_layers=8,
            num_cls_layers=2,
            block_params=None,
            cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
            fc_params=[],
            activation='gelu',
            # misc
            trim=True,
        for_inference=False,
    )
    cfg.update(**kwargs)
    _logger.info('Model config: %s' % str(cfg))

    model = ParticleTransformerWrapper(**cfg)

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax': {0: 'N'}}},
    }

    return model, model_info

class ResNetUnit(nn.Module):
    r"""Parameters
    ----------
    in_channels : int
        Number of channels in the input vectors.
    out_channels : int
        Number of channels in the output vectors.
    strides: tuple
        Strides of the two convolutional layers, in the form of (stride0, stride1)
    """

    def __init__(self, in_channels, out_channels, strides=(1, 1), **kwargs):

        super(ResNetUnit, self).__init__(**kwargs)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=strides[0], padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=strides[1], padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dim_match = True
        if not in_channels == out_channels or not strides == (1, 1):  # dimensions not match
            self.dim_match = False
            self.conv_sc = nn.Conv1d(in_channels, out_channels, kernel_size=1,
                                     stride=strides[0] * strides[1], bias=False)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # print('resnet unit', identity.shape, x.shape, self.dim_match)
        if self.dim_match:
            return identity + x
        else:
            return self.conv_sc(identity) + x


class ResNet(nn.Module):
    r"""Parameters
    ----------
    features_dims : int
        Input feature dimensions.
    num_classes : int
        Number of output classes.
    conv_params : list
        List of the convolution layer parameters. 
        The first element is a tuple of size 1, defining the transformed feature size for the initial feature convolution layer.
        The following are tuples of feature size for multiple stages of the ResNet units. Each number defines an individual ResNet unit.
    fc_params: list
        List of fully connected layer parameters after all EdgeConv blocks, each element in the format of
        (n_feat, drop_rate)
    """

    def __init__(self, features_dims, num_classes,
                 conv_params=[(32,), (64, 64), (64, 64), (128, 128)],
                 fc_params=[(512, 0.2)],
                 for_inference=False,
                 **kwargs):

        super(ResNet, self).__init__(**kwargs)
        self.conv_params = conv_params
        self.num_stages = len(conv_params) - 1
        self.fts_conv = nn.Sequential(
            nn.BatchNorm1d(features_dims),
            nn.Conv1d(
                in_channels=features_dims, out_channels=conv_params[0][0],
                kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(conv_params[0][0]),
            nn.ReLU())

        # define ResNet units for each stage. Each unit is composed of a sequence of ResNetUnit block
        self.resnet_units = nn.ModuleDict()
        for i in range(self.num_stages):
            # stack units[i] layers in this stage
            unit_layers = []
            for j in range(len(conv_params[i + 1])):
                in_channels, out_channels = (conv_params[i][-1], conv_params[i + 1][0]) if j == 0 \
                    else (conv_params[i + 1][j - 1], conv_params[i + 1][j])
                strides = (2, 1) if (j == 0 and i > 0) else (1, 1)
                unit_layers.append(ResNetUnit(in_channels, out_channels, strides))

            self.resnet_units.add_module('resnet_unit_%d' % i, nn.Sequential(*unit_layers))

        # define fully connected layers
        fcs = []
        for idx, layer_param in enumerate(fc_params):
            channels, drop_rate = layer_param
            in_chn = conv_params[-1][-1] if idx == 0 else fc_params[idx - 1][0]
            fcs.append(nn.Sequential(nn.Linear(in_chn, channels), nn.ReLU(), nn.Dropout(drop_rate)))
        fcs.append(nn.Linear(fc_params[-1][0], num_classes))
        if for_inference:
            fcs.append(nn.Softmax(dim=1))
        self.fc = nn.Sequential(*fcs)

    def forward(self, points, features, lorentz_vectors, mask):
        # x: the feature vector, (N, C, P)
        if mask is not None:
            features = features * mask
        x = self.fts_conv(features)
        for i in range(self.num_stages):
            x = self.resnet_units['resnet_unit_%d' % i](x)  # (N, C', P'), P'<P due to kernal_size>1 or stride>1

        # global average pooling
        x = x.mean(dim=-1)  # (N, C')
        # fully connected
        x = self.fc(x)  # (N, out_chn)
        return x


def get_model(data_config, **kwargs):
    conv_params = [(32,), (64, 64), (64, 64), (128, 128)]
    fc_params = [(512, 0.2)]

    pf_features_dims = len(data_config.input_dicts['pf_features'])
    num_classes = len(data_config.label_value)
    model = ResNet(pf_features_dims, num_classes,
                   conv_params=conv_params,
                   fc_params=fc_params)

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax': {0: 'N'}}},
    }

    return model, model_info

def get_loss(data_config, **kwargs):
    distil_weight = kwargs.get('distill_weight', 0.9)
    temp = kwargs.get('temp', 10.)
    loss_fn_name = kwargs.get('loss_fn', 'KL_div')
    if loss_fn_name == 'cross_entropy':
        loss_fn = nn.CrossEntropyLoss()
    elif loss_fn_name == 'MSE':
        loss_fn = nn.MSELoss()
    elif loss_fn_name == 'KL_div':
        loss_fn = nn.KLDivLoss()
    def kd_loss_fn(model_output, teacher_model_output,label):
        soft_teacher_out = F.softmax(teacher_model_output / temp, dim=1)
        soft_student_out = F.softmax(model_output / temp, dim=1)

        loss = (1 - distil_weight) * F.cross_entropy(model_output, label)
        loss += (distil_weight * temp * temp) * loss_fn(
            soft_teacher_out, soft_student_out
        )
        return loss
    return kd_loss_fn 