from typing import Optional, Tuple

import torch as th
import torch.nn as nn


class DWSConvLSTM2d(nn.Module):
    """LSTM with (depthwise-separable) Conv option in NCHW [channel-first] format.
    """

    def __init__(self,
                 dim: int,
                 dws_conv: bool = True,
                 dws_conv_only_hidden: bool = True,
                 dws_conv_kernel_size: int = 3,
                 cell_update_dropout: float = 0.):
        super().__init__()
        assert isinstance(dws_conv, bool)
        assert isinstance(dws_conv_only_hidden, bool)
        self.dim = dim

        xh_dim = dim * 2
        gates_dim = dim * 4
        conv3x3_dws_dim = dim if dws_conv_only_hidden else xh_dim
        self.conv3x3_dws = nn.Conv2d(in_channels=conv3x3_dws_dim,
                                     out_channels=conv3x3_dws_dim,
                                     kernel_size=dws_conv_kernel_size,
                                     padding=dws_conv_kernel_size // 2,
                                     groups=conv3x3_dws_dim) if dws_conv else nn.Identity()
        self.conv1x1 = nn.Conv2d(in_channels=xh_dim,
                                 out_channels=gates_dim,
                                 kernel_size=1)
        self.conv_only_hidden = dws_conv_only_hidden
        self.cell_update_dropout = nn.Dropout(p=cell_update_dropout)

    def forward(self, x: th.Tensor, h_and_c_previous: Optional[Tuple[th.Tensor, th.Tensor]] = None) \
            -> Tuple[th.Tensor, th.Tensor]:
        """
        :param x: (N C H W)
        :param h_and_c_previous: ((N C H W), (N C H W))
        :return: ((N C H W), (N C H W))
        """
        if h_and_c_previous is None:
            # generate zero states
            hidden = th.zeros_like(x)
            cell = th.zeros_like(x)
            h_and_c_previous = (hidden, cell)
        h_tm1, c_tm1 = h_and_c_previous

        if self.conv_only_hidden:
            h_tm1 = self.conv3x3_dws(h_tm1)
        xh = th.cat((x, h_tm1), dim=1)
        if not self.conv_only_hidden:
            xh = self.conv3x3_dws(xh)
        mix = self.conv1x1(xh)

        gates, cell_input = th.tensor_split(mix, [self.dim * 3], dim=1)
        assert gates.shape[1] == cell_input.shape[1] * 3

        gates = th.sigmoid(gates)
        forget_gate, input_gate, output_gate = th.tensor_split(gates, 3, dim=1)
        assert forget_gate.shape == input_gate.shape == output_gate.shape

        cell_input = self.cell_update_dropout(th.tanh(cell_input))

        c_t = forget_gate * c_tm1 + input_gate * cell_input
        h_t = output_gate * th.tanh(c_t)

        return h_t, c_t


class DWSConvSTLSTM2d(nn.Module):
    """Spatiotemporal LSTM (ST-LSTM) from PredRNN with optional depthwise-separable Conv.

    Compared to standard ConvLSTM, ST-LSTM maintains two memory cells:
      - C: temporal memory, updated via (x, h_{t-1}), same as ConvLSTM
      - M: spatial memory, updated via (x, m_{t-1}), propagated temporally within each stage
    The hidden state is derived from both memories:
      H_t = o_t * tanh(W_1x1([C_t, M_t]))

    When zigzag=false, M is carried across time within each stage independently.
    When zigzag=true, M flows across layers within each time step (l-1 -> l),
    and from the last layer at t-1 to the first layer at t (zigzag connection).
    """

    def __init__(self,
                 dim: int,
                 dws_conv: bool = True,
                 dws_conv_only_hidden: bool = True,
                 dws_conv_kernel_size: int = 3,
                 cell_update_dropout: float = 0.):
        super().__init__()
        assert isinstance(dws_conv, bool)
        assert isinstance(dws_conv_only_hidden, bool)
        self.dim = dim

        xh_dim = dim * 2

        # Depthwise-separable conv on h_{t-1} (reuse original design)
        conv3x3_dws_dim = dim if dws_conv_only_hidden else xh_dim
        self.conv3x3_dws = nn.Conv2d(
            in_channels=conv3x3_dws_dim,
            out_channels=conv3x3_dws_dim,
            kernel_size=dws_conv_kernel_size,
            padding=dws_conv_kernel_size // 2,
            groups=conv3x3_dws_dim) if dws_conv else nn.Identity()
        self.conv_only_hidden = dws_conv_only_hidden

        # Temporal memory path: cat(x, h_{t-1}) -> i, f, g  (3 * dim)
        self.conv1x1_temporal = nn.Conv2d(in_channels=xh_dim, out_channels=dim * 3, kernel_size=1)

        # Spatial memory path: cat(x, m_{t-1}) -> i', f', g'  (3 * dim)
        self.conv1x1_spatial = nn.Conv2d(in_channels=xh_dim, out_channels=dim * 3, kernel_size=1)

        # Output gate: cat(x, h_{t-1}, C_t, M_t) -> o  (dim)
        self.conv1x1_output = nn.Conv2d(in_channels=dim * 4, out_channels=dim, kernel_size=1)

        # Combine C and M for hidden state: cat(C_t, M_t) -> dim
        self.conv1x1_memory = nn.Conv2d(in_channels=dim * 2, out_channels=dim, kernel_size=1)

        self.cell_update_dropout = nn.Dropout(p=cell_update_dropout)

    def forward(self, x: th.Tensor,
                h_and_c_previous: Optional[Tuple[th.Tensor, th.Tensor]] = None,
                m_previous: Optional[th.Tensor] = None) \
            -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        :param x: (N C H W)
        :param h_and_c_previous: ((N C H W), (N C H W)) temporal state, or None
        :param m_previous: (N C H W) spatial memory from zigzag/cross-layer routing, or None
        :return: (h_t, c_t, m_t), each (N C H W)
        """
        if h_and_c_previous is None:
            h_tm1 = th.zeros_like(x)
            c_tm1 = th.zeros_like(x)
        else:
            h_tm1, c_tm1 = h_and_c_previous

        m_tm1 = m_previous if m_previous is not None else th.zeros_like(x)

        # --- Temporal memory C update (same gating as ConvLSTM) ---
        if self.conv_only_hidden:
            h_tm1_conv = self.conv3x3_dws(h_tm1)
        else:
            h_tm1_conv = h_tm1

        xh = th.cat((x, h_tm1_conv), dim=1)
        if not self.conv_only_hidden:
            xh = self.conv3x3_dws(xh)

        temporal_mix = self.conv1x1_temporal(xh)  # (N, 3*dim, H, W)
        t_i, t_f, t_g = th.tensor_split(temporal_mix, 3, dim=1)
        t_i = th.sigmoid(t_i)
        t_f = th.sigmoid(t_f)
        t_g = self.cell_update_dropout(th.tanh(t_g))
        c_t = t_f * c_tm1 + t_i * t_g

        # --- Spatial memory M update ---
        xm = th.cat((x, m_tm1), dim=1)
        spatial_mix = self.conv1x1_spatial(xm)  # (N, 3*dim, H, W)
        s_i, s_f, s_g = th.tensor_split(spatial_mix, 3, dim=1)
        s_i = th.sigmoid(s_i)
        s_f = th.sigmoid(s_f)
        s_g = self.cell_update_dropout(th.tanh(s_g))
        m_t = s_f * m_tm1 + s_i * s_g

        # --- Output gate (combines all four sources) ---
        o_t = th.sigmoid(self.conv1x1_output(th.cat((x, h_tm1, c_t, m_t), dim=1)))

        # --- Hidden state from combined memories ---
        h_t = o_t * th.tanh(self.conv1x1_memory(th.cat((c_t, m_t), dim=1)))

        return h_t, c_t, m_t
