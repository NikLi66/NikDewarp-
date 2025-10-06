import torch
import torch.nn as nn
import torch.nn.functional as F

class msca_attention(nn.Module):

    def __init__(self, channels, groups=1):
        super(msca_attention, self).__init__()
        self.conv0 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels)
        self.conv0_1 = nn.Conv2d(channels, channels, kernel_size=(1, 7), padding=(0, 3), groups=channels)
        self.conv0_2 = nn.Conv2d(channels, channels, kernel_size=(7, 1), padding=(3, 0), groups=channels)
        self.conv1_1 = nn.Conv2d(channels, channels, kernel_size=(1, 11), padding=(0, 5), groups=channels)
        self.conv1_2 = nn.Conv2d(channels, channels, kernel_size=(11, 1), padding=(5, 0), groups=channels)
        self.conv2_1 = nn.Conv2d(channels, channels, kernel_size=(1, 21), padding=(0, 10), groups=channels)
        self.conv2_2 = nn.Conv2d(channels, channels, kernel_size=(21, 1), padding=(10, 0), groups=channels)
        self.conv3 = nn.Conv2d(channels, channels, 1, groups=groups)

    def forward(self, x):
        """Forward function."""
        # x shape[b, hw, c]

        u = x.clone()

        attn = self.conv0(x)

        # Multi-Scale Feature extraction
        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)

        attn =  attn + attn_0 + attn_1 + attn_2
        # Channel Mixing
        attn = self.conv3(attn)

        # Convolutional Attention
        x = attn * u

        return x

class msca_input_layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(msca_input_layer, self).__init__()

        self._in_channels = in_channels
        self._out_channels = out_channels

        self._conv1 = nn.Sequential(
            self._bn_fn(self._in_channels),
            nn.Conv2d(self._in_channels, self._out_channels, stride=4, groups=2),
            self._bn_fn(self._out_channels),
            self._act_fn(inplace=True)
        )
        self._conv2 = nn.Sequential(
            msca_attention(self._out_channels),
            self._bn_fn(self._out_channels),
            self._act_fn(inplace=True)
        )

    def forward(self, x):
        # self._logger.info("layer_input in x {}".format(x.shape))
        x = self._conv1(x)
        x = self._conv2(x)
        # self._logger.info("layer_input out x {}".format(x.shape))
        return x

class msca_down_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(msca_down_layer, self).__init__()

        self._in_channels = in_channels
        self._out_channels = out_channels

        self._conv = nn.Sequential(
            nn.Conv2d(self._in_channels, self._out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2),
            nn.InstanceNorm2d(self._out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """Forward function."""

        x = self._conv(x)
        return x


class msca_layer(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.in_channels = channels

        self.num_units = channels * 4

        #msca
        self.msca = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            msca_attention(channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        # self.msca_fc1 = nn.Conv2d(channels, channels, kernel_size=1),
        # self.msca_act_fn = nn.ReLU(inplace=True)
        # self.msca_norm = nn.InstanceNorm2d(channels)
        # self.msca_att = msca_attention(channels)
        # self.msca_fc2 = nn.Conv2d(channels, channels, kernel_size=1)
        # self.msca_norm = nn.InstanceNorm2d(channels)

        #ffn
        # self.ffn_norm = nn.InstanceNorm2d(channels)
        # self.ffn_fc1 = nn.Conv2d(channels, self.num_units, kernel_size=1)
        # self.ffn_dwconv = nn.Conv2d( self.num_units, self.num_units, 3, 1, 1, bias=True, groups=self.num_units)
        # self.ffn_act_fn = nn.ReLU(inplace=True)
        # self.ffn_fc2 = nn.Conv2d(self.num_units, channels, kernel_size=1)

    def forward(self, x):
        """Forward function."""

        return  x + self.msca(x)

        #msca
        # o1 = self.msca_norm(x)
        # o1 = self.msca_fc1(o1)
        # o1 = self.msca_act_fn(o1)
        # o1 = self.msca_att(o1)
        # o1 = self.msca_fc2(o1)
        # o1 = x + o1

        # #ffn
        # o2 = self.ffn_norm(o1)
        # o2 = self.ffn_fc1(o2)
        # o2 = self.ffn_dwconv(o2)
        # o2 = self.ffn_act_fn(o2)
        # o2 = self.ffn_fc2(o2)
        #
        # #output
        # o = o1 + o2

        # return o1


class cross_attention(nn.Module):
    def __init__(self,  num_units, num_heads=8, dropout_rate=0):
        """Applies multihead attention.

        Args:
            num_units: A scalar. Attention size.
            dropout_rate: A floating point number.
            causality: Boolean. If true, units that reference the future are masked.
            num_heads: An int. Number of heads.
        """
        super(cross_attention, self).__init__()

        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

    def forward(self, queries, keys, values):
        # keys, values: same shape of [N, T_k, C_k]
        # queries: A 3d Variable with shape of [N, T_q, C_q]
        # [b,hw,c]

        # Split and concat
        Q_ = torch.cat(torch.chunk(queries, self.num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)
        K_ = torch.cat(torch.chunk(keys, self.num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)
        V_ = torch.cat(torch.chunk(values, self.num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)

        # Multiplication - batch matrix multiply
        outputs = torch.bmm(Q_, K_.permute(0, 2, 1))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.size()[-1] ** 0.5)

        # Activation
        outputs = F.softmax(outputs, dim=-1)  # (h*N, T_q, T_k)

        # Dropouts
        # outputs = self.output_dropout(outputs)  # (h*N, T_q, T_k)

        # Weighted sum
        outputs = torch.bmm(outputs, V_)  # (h*N, T_q, C/h)

        # Restore shape
        outputs = torch.cat(
            torch.chunk(outputs, self.num_heads, dim=0), dim=2
        )  # (N, T_q, C)

        return outputs

class feedformer_layer(nn.Module):

    def __init__(self, q_dim, kv_dim, num_heads):
        super().__init__()

        #att
        num_units = q_dim
        self.att_norm_q = nn.LayerNorm(q_dim)
        self.att_norm_kv = nn.LayerNorm(kv_dim)
        self.att_fc_q = nn.Linear(q_dim, q_dim)
        self.att_fc_k = nn.Linear(kv_dim, num_units)
        self.att_fc_v = nn.Linear(kv_dim, num_units)
        self.att_attention = cross_attention(num_units=q_dim, num_heads=num_heads)

        #ffn
        self.mlp_hidden_dim = q_dim * 4
        self.ffn_norm = nn.InstanceNorm2d(q_dim)
        self.ffn_fc1 = nn.Linear(q_dim, self.mlp_hidden_dim)
        self.ffn_dwconv = nn.Conv2d(self.mlp_hidden_dim, self.mlp_hidden_dim, 3, 1, 1, bias=True, groups=self.mlp_hidden_dim)
        self.ffn_act_fn = nn.GELU()
        self.ffn_fc2 = nn.Linear(self.mlp_hidden_dim, q_dim)

    def forward(self, q, q_h, q_w, kv):
        #[B, HW, C]

        # attention
        q1 = self.att_norm_q(q)
        kv = self.att_norm_kv(kv)
        q1 = self.att_fc_q(q1)
        k1 = self.att_fc_k(kv)
        v1 = self.att_fc_v(kv)
        o1 = self.att_attention(q1, k1, v1)
        o1 = q + o1

        # ffn
        o2 = self.ffn_norm(o1)
        o2 = self.ffn_fc1(o2)
        o2 = o2.transpose(1, 2).reshape(-1, self.mlp_hidden_dim, q_h, q_w)
        o2 = self.ffn_dwconv(o2)
        o2 = o2.flatten(2).transpose(1, 2)
        o2 = self.ffn_act_fn(o2)
        o2 = self.ffn_fc2(o2)

        #output
        o = o1 + o2

        return o
