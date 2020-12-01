import torch
import torch.nn as nn
from typing import List
import torch.nn.functional as F
from config import config
from enum import Enum

activation_func = F.relu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_units = config.num_units


class SpectrumEncoding(nn.Module):
    """
    every ops in this module is performed on CPU
    """
    def __init__(self, d_model, max_len, spectrum_reso):
        super(SpectrumEncoding, self).__init__()
        self.d_model = d_model
        self.spectrum_reso = spectrum_reso
        self.N = config.MAX_NUM_PEAK
        pe = torch.zeros(max_len + 1, d_model)
        position = torch.arange(0, max_len + 1, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe[0, :] = 0.
        self.register_buffer('pe', pe)

    def forward(self, peaks_location, peaks_intensity):
        """
        peaks_location: [batch, N]
        peaks_intensity: [batch, N]
        return: [batch, embed_size], embedded spectrum representation
        """
        cpu_location = peaks_location.to(torch.device("cpu"))
        cpu_intensity = peaks_intensity.to(torch.device("cpu"))
        batch = peaks_location.size(0)
        batch_embedded_spectrum = torch.zeros(batch, self.d_model)
        loc = torch.ceil(cpu_location * self.spectrum_reso).long()
        for i in range(self.N):
            temp1 = torch.index_select(self.pe, dim=0, index=loc[:, i])
            temp2 = temp1 * cpu_intensity[:, i:i+1]
            batch_embedded_spectrum += temp2
        return batch_embedded_spectrum


class TNet(nn.Module):
    """
    the T-net structure in the Point Net paper
    """

    def __init__(self, with_lstm=False):
        super(TNet, self).__init__()
        self.with_lstm = with_lstm
        self.conv1 = nn.Conv1d(config.vocab_size * config.num_ion + 1, num_units, 1)
        self.conv2 = nn.Conv1d(num_units, 2 * num_units, 1)
        self.conv3 = nn.Conv1d(2 * num_units, 4 * num_units, 1)
        self.fc1 = nn.Linear(4 * num_units, 2 * num_units)
        self.fc2 = nn.Linear(2 * num_units, num_units)
        if not with_lstm:
            self.output_layer = nn.Linear(num_units, config.vocab_size)
        self.relu = nn.ReLU()

        self.input_batch_norm = nn.BatchNorm1d(config.vocab_size * config.num_ion + 1)

        self.bn1 = nn.BatchNorm1d(num_units)
        self.bn2 = nn.BatchNorm1d(2 * num_units)
        self.bn3 = nn.BatchNorm1d(4 * num_units)
        self.bn4 = nn.BatchNorm1d(2 * num_units)
        self.bn5 = nn.BatchNorm1d(num_units)

    def forward(self, x):
        """

        :param x: [batch * T, 26*8+1, N]
        :return:
            logit: [batch * T, 26]
        """
        x = self.input_batch_norm(x)
        x = activation_func(self.bn1(self.conv1(x)))
        x = activation_func(self.bn2(self.conv2(x)))
        x = activation_func(self.bn3(self.conv3(x)))
        x, _ = torch.max(x, dim=2)  # global max pooling

        x = activation_func(self.bn4(self.fc1(x)))
        x = activation_func(self.bn5(self.fc2(x)))
        if not self.with_lstm:
            x = self.output_layer(x)  # [batch * T, 26]
        return x


class DeepNovoPointNet(nn.Module):
    def __init__(self):
        super(DeepNovoPointNet, self).__init__()
        self.t_net = TNet(with_lstm=False)
        self.distance_scale_factor = nn.Parameter(torch.tensor(0).float(), requires_grad=True)

    def forward(self, location_index, peaks_location, peaks_intensity):
        """

        :param location_index: [batch, T, 26, 8] float
        :param peaks_location: [batch, N] N stands for MAX_NUM_PEAK, float
        :param peaks_intensity: [batch, N], float32
        :return:
            logits: [batch, T, 26]
        """

        N = peaks_location.size(1)
        batch_size, T, vocab_size, num_ion = location_index.size()

        peaks_location = peaks_location.view(batch_size, 1, N, 1)
        peaks_intensity = peaks_intensity.view(batch_size, 1, N, 1)
        peaks_location = peaks_location.expand(-1, T, -1, -1)  # [batch, T, N, 1]
        peaks_location_mask = (peaks_location > 1e-5).float()
        peaks_intensity = peaks_intensity.expand(-1, T, -1, -1)  # [batch, T, N, 1]

        location_index = location_index.view(batch_size, T, 1, vocab_size * num_ion)
        location_index_mask = (location_index > 1e-5).float()

        ppm_diff = (peaks_location - location_index) / (location_index + 1e-6) * 1e6

        location_exp_minus_abs_diff = torch.exp(
            -torch.abs(
                ppm_diff / 5 * torch.sigmoid(self.distance_scale_factor)
            )
        )
        # [batch, T, N, 26*8]

        location_exp_minus_abs_diff = location_exp_minus_abs_diff * peaks_location_mask * location_index_mask

        input_feature = torch.cat((location_exp_minus_abs_diff, peaks_intensity), dim=3)
        input_feature = input_feature.view(batch_size * T, N, vocab_size * num_ion + 1)
        input_feature = input_feature.transpose(1, 2)

        result = self.t_net(input_feature).view(batch_size, T, vocab_size)
        return result


class InitNet(nn.Module):
    def __init__(self):
        super(InitNet, self).__init__()
        self.init_state_layer = nn.Linear(config.embedding_size, 2 * config.lstm_hidden_units)
        self.lstm_hidden_units = config.lstm_hidden_units
        self.num_lstm_layers = config.num_lstm_layers

    def forward(self, spectrum_representation):
        """

        :param spectrum_representation: [batch, embedding_size]
        :return:
            [num_lstm_layers, batch_size, lstm_units], [num_lstm_layers, batch_size, lstm_units],
        """
        x = torch.tanh(self.init_state_layer(spectrum_representation))
        h_0, c_0 = torch.split(x, self.lstm_hidden_units, dim=1)
        h_0 = torch.unsqueeze(h_0, dim=0)
        h_0 = h_0.repeat(self.num_lstm_layers, 1, 1)
        c_0 = torch.unsqueeze(c_0, dim=0)
        c_0 = c_0.repeat(self.num_lstm_layers, 1, 1)
        return h_0, c_0


class DeepNovoPointNetWithLSTM(nn.Module):
    def __init__(self):
        super(DeepNovoPointNetWithLSTM, self).__init__()
        self.t_net = TNet(with_lstm=True)
        self.embedding = nn.Embedding(num_embeddings=config.vocab_size,
                                      embedding_dim=config.embedding_size)
        self.lstm = nn.LSTM(config.embedding_size,
                            config.lstm_hidden_units,
                            num_layers=config.num_lstm_layers,
                            batch_first=True)
        # make sure lstm module has contiguous weights
        self.lstm.flatten_parameters()
        self.dropout = nn.Dropout(config.dropout_rate)
        self.output_layer = nn.Linear(config.num_units + config.lstm_hidden_units,
                                      config.vocab_size)
        self.distance_scale_factor = nn.Parameter(torch.tensor(0).float(), requires_grad=True)


    def forward(self, location_index, peaks_location, peaks_intensity, aa_input, state_tuple):
        """

        :param location_index: [batch, T, 26, 8] long
        :param peaks_location: [batch, N] N stands for MAX_NUM_PEAK, long
        :param peaks_intensity: [batch, N], float32
        :param aa_input:[batch, T]
        :param state_tuple: (h0, c0), where each is [num_lstm_layer, batch_size, num_units] tensor
        :return:
            logits: [batch, T, 26]
        """
        self.lstm.flatten_parameters()
        N = peaks_location.size(1)
        batch_size, T, vocab_size, num_ion = location_index.size()

        peaks_location = peaks_location.view(batch_size, 1, N, 1)
        peaks_intensity = peaks_intensity.view(batch_size, 1, N, 1)
        peaks_location = peaks_location.expand(-1, T, -1, -1)  # [batch, T, N, 1]
        peaks_location_mask = (peaks_location > 1e-5).float()
        peaks_intensity = peaks_intensity.expand(-1, T, -1, -1)  # [batch, T, N, 1]

        location_index = location_index.view(batch_size, T, 1, vocab_size * num_ion)
        location_index_mask = (location_index > 1e-5).float()

        ppm_diff = (peaks_location - location_index) / (location_index + 1e-6) * 1e6

        location_exp_minus_abs_diff = torch.exp(
            -torch.abs(
                ppm_diff / 5 * torch.sigmoid(self.distance_scale_factor)
            )
        )
        # [batch, T, N, 26*8]

        location_exp_minus_abs_diff = location_exp_minus_abs_diff * peaks_location_mask * location_index_mask

        input_feature = torch.cat((location_exp_minus_abs_diff, peaks_intensity), dim=3)
        input_feature = input_feature.view(batch_size * T, N, vocab_size * num_ion + 1)
        input_feature = input_feature.transpose(1, 2)

        ion_feature = self.t_net(input_feature).view(batch_size, T, config.num_units)  # attention on peaks

        # embedding
        aa_embedded = self.embedding(aa_input)
        lstm_input = aa_embedded  # [batch, T, embedding_size]
        # lstm_input = self.dropout(lstm_input)
        output_feature, new_state_tuple = self.lstm(lstm_input, state_tuple)
        output_feature = torch.cat((ion_feature, activation_func(output_feature)), dim=2)
        output_feature = self.dropout(output_feature)
        logit = self.output_layer(output_feature)
        return logit, new_state_tuple


if config.use_lstm:
    DeepNovoModel = DeepNovoPointNetWithLSTM
else:
    DeepNovoModel = DeepNovoPointNet


class Direction(Enum):
    forward = 1
    backward = 2


class InferenceModelWrapper(object):
    """
    a wrapper class so that the beam search part of code is the same for both with lstm and without lstm model.
    TODO(Rui): support no lstm branch here
    """

    def __init__(self, forward_model: DeepNovoModel, backward_model: DeepNovoModel, init_net: InitNet = None):
        self.forward_model = forward_model
        self.backward_model = backward_model
        # make sure all models are in eval mode
        self.forward_model.eval()
        self.backward_model.eval()
        self.spectrum_encoding = SpectrumEncoding(d_model=config.embedding_size, max_len=config.n_position,
                                                  spectrum_reso=config.spectrum_reso)
        if config.use_lstm:
            assert init_net is not None
            self.init_net = init_net
            self.init_net.eval()

    def step(self, candidate_location, peaks_location, peaks_intensity, aa_input, state_tuple, direction):
        """
        :param state_tuple: tuple of ([num_layer, batch_size, num_unit], [num_layer, batch_size, num_unit])
        :param aa_input: [batch, 1]
        :param candidate_location: [batch, 1, 26, 8]
        :param peaks_location: [batch, N]
        :param peaks_intensity: [batch, N]
        :param direction: enum class, whether forward or backward
        :return: (log_prob, new_hidden_state)
        log_prob: the pred log prob of shape [batch, 26]
        """
        if direction == Direction.forward:
            model = self.forward_model
        else:
            model = self.backward_model

        with torch.no_grad():
            if config.use_lstm:
                logit, new_state_tuple = model(candidate_location, peaks_location, peaks_intensity, aa_input,
                                               state_tuple)
            else:
                logit = model(candidate_location, peaks_location, peaks_intensity)
                new_state_tuple = None
            logit = torch.squeeze(logit, dim=1)
            log_prob = F.log_softmax(logit)
            # log_prob = F.logsigmoid(logit)
        return log_prob, new_state_tuple

    def initial_hidden_state(self, spectrum_representation):
        """

        :param: spectrum_representation, [batch, embedding_size]
        :return:
            [num_lstm_layers, batch_size, lstm_units], [num_lstm_layers, batch_size, lstm_units],
        """
        with torch.no_grad():
            h_0, c_0 = self.init_net(spectrum_representation)
            return h_0.to(device), c_0.to(device)


# _get_ion_index_device = torch.device("cpu")
_get_ion_index_device = device
mass_ID_torch = torch.from_numpy(config.mass_ID_np).to(_get_ion_index_device).unsqueeze(0)
mass_CO = config.mass_CO
mass_H2O = config.mass_H2O
mass_NH3 = config.mass_NH3
mass_H = config.mass_H
MZ_MAX = config.MZ_MAX


def torch_get_ion_index(peptide_mass: List[float], prefix_mass: List[float], direction):
    """

    :param peptide_mass: neutral mass of a peptide
    :param prefix_mass:
    :param direction: 0 for forward, 1 for backward
    :return: an int32 ndarray of shape [26, 8], each element represent a index of the spectrum embbeding matrix. for out
    of bound position, the index is 0
    """
    peptide_mass = torch.tensor(peptide_mass, dtype=torch.float32, device=_get_ion_index_device).unsqueeze(1)  # batch
    prefix_mass = torch.tensor(prefix_mass, dtype=torch.float32, device=_get_ion_index_device).unsqueeze(1)  # batch
    with torch.no_grad():
        if direction == 0:
            candidate_b_mass = prefix_mass + mass_ID_torch  # [batch, 26]
            candidate_y_mass = peptide_mass - candidate_b_mass
        else:
            candidate_y_mass = prefix_mass + mass_ID_torch
            candidate_b_mass = peptide_mass - candidate_y_mass
        candidate_a_mass = candidate_b_mass - mass_CO

        # b-ions
        candidate_b_H2O = candidate_b_mass - mass_H2O
        candidate_b_NH3 = candidate_b_mass - mass_NH3
        candidate_b_plus2_charge1 = ((candidate_b_mass + 2 * mass_H) / 2
                                     - mass_H)

        # a-ions
        candidate_a_H2O = candidate_a_mass - mass_H2O
        candidate_a_NH3 = candidate_a_mass - mass_NH3
        candidate_a_plus2_charge1 = ((candidate_a_mass + 2 * mass_H) / 2
                                     - mass_H)

        # y-ions
        candidate_y_H2O = candidate_y_mass - mass_H2O
        candidate_y_NH3 = candidate_y_mass - mass_NH3
        candidate_y_plus2_charge1 = ((candidate_y_mass + 2 * mass_H) / 2
                                     - mass_H)

        # ion_2
        # ~   b_ions = [candidate_b_mass]
        # ~   y_ions = [candidate_y_mass]
        # ~   ion_mass_list = b_ions + y_ions

        ion_mass_list = [candidate_b_mass,
                         candidate_b_H2O,
                         candidate_b_NH3,
                         candidate_b_plus2_charge1,
                         candidate_y_mass,
                         candidate_y_H2O,
                         candidate_y_NH3,
                         candidate_y_plus2_charge1,
                         candidate_a_mass,
                         candidate_a_H2O,
                         candidate_a_NH3,
                         candidate_a_plus2_charge1]
        ion_mass = torch.stack(ion_mass_list, dim=2)  # [batch, 26, 12]

        in_bound_mask = torch.logical_and(
            ion_mass > 0,
            ion_mass <= MZ_MAX).float()
        ion_location = ion_mass * in_bound_mask  # [batch, 26, 12], out of bound index would have value 0
    return ion_location.unsqueeze(1).contiguous()  # [batch, 1, 26, 12]
