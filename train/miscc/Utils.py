from Config import config
cfg = config()

import errno
import os

import numpy as np
import scipy.signal
import torch

from train.Model.MESH_encoder.MESH_model import SceneEncoderModel
from train.Model.RIR_decoder.Dec_model import SRIRParameterDecoder


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def _strip_module_prefix(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k.replace('module.', '')] = v
    return new_state_dict


def load_model_train(model_cfg):
    model_dir = model_cfg.DATA.out_path
    scene_encoder = SceneEncoderModel(model_cfg)
    with open(os.path.join(model_dir, 'Mesh_Enc.txt'), 'w', encoding='utf-8') as f:
        print(scene_encoder, file=f)

    srir_decoder = SRIRParameterDecoder(model_cfg)
    with open(os.path.join(model_dir, 'RIR_G.txt'), 'w', encoding='utf-8') as f:
        print(srir_decoder, file=f)

    mesh_checkpoint_path = os.path.join(model_dir, 'Model', model_cfg.TRAIN.MESH_NET)
    if model_cfg.TRAIN.MESH_NET:
        state_dict = torch.load(mesh_checkpoint_path, map_location=lambda storage, loc: storage)
        scene_encoder.load_state_dict(_strip_module_prefix(state_dict))
        print('Load from:', mesh_checkpoint_path)

    decoder_checkpoint_path = os.path.join(model_dir, 'Model', model_cfg.TRAIN.NET_G)
    if model_cfg.TRAIN.NET_G:
        state_dict = torch.load(decoder_checkpoint_path, map_location=lambda storage, loc: storage)
        srir_decoder.load_state_dict(state_dict)
        print('Load from:', decoder_checkpoint_path)

    return scene_encoder, srir_decoder, None


def load_model_test(model_cfg):
    model_dir = model_cfg.DATA.out_path
    scene_encoder = SceneEncoderModel(mesh_cfg=model_cfg)
    srir_decoder = SRIRParameterDecoder(model_cfg=model_cfg)

    mesh_checkpoint_path = os.path.join(model_dir, 'Model', model_cfg.TEST.MESH_NET)
    if model_cfg.TEST.MESH_NET:
        state_dict = torch.load(mesh_checkpoint_path, map_location=lambda storage, loc: storage)
        scene_encoder.load_state_dict(_strip_module_prefix(state_dict))
        print('Load from:', mesh_checkpoint_path)

    decoder_checkpoint_path = os.path.join(model_dir, 'Model', model_cfg.TEST.NET_G)
    if model_cfg.TEST.NET_G:
        state_dict = torch.load(decoder_checkpoint_path, map_location=lambda storage, loc: storage)
        srir_decoder.load_state_dict(state_dict)
        print('Load from:', decoder_checkpoint_path)

    return scene_encoder, srir_decoder


def save_model(Mesh_Enc, RIR_G, RIR_D, epoch, model_dir):
    if Mesh_Enc is not None:
        torch.save(Mesh_Enc.state_dict(), f'{model_dir}/Mesh_Enc_epoch_{epoch}.pth')
    if RIR_G is not None:
        torch.save(RIR_G.state_dict(), f'{model_dir}/RIR_G_epoch_{epoch}.pth')
    if RIR_D is not None:
        torch.save(RIR_D.state_dict(), f'{model_dir}/RIR_D_epoch_{epoch}.pth')


def save_newest_model(Mesh_Enc, RIR_G, RIR_D, model_dir, epoch):
    epoch_tag = str(epoch).zfill(4)
    if Mesh_Enc is not None:
        torch.save(Mesh_Enc.state_dict(), f'{model_dir}/Mesh_Enc_epoch_{epoch_tag}.pth')
    if RIR_G is not None:
        torch.save(RIR_G.state_dict(), f'{model_dir}/RIR_G_epoch_{epoch_tag}.pth')
    if RIR_D is not None:
        torch.save(RIR_D.state_dict(), f'{model_dir}/RIR_D_epoch_{epoch_tag}.pth')

    stale_epoch_tag = str(epoch - 4).zfill(4)
    stale_mesh_path = f'{model_dir}/Mesh_Enc_epoch_{stale_epoch_tag}.pth'
    if os.path.exists(stale_mesh_path):
        os.remove(stale_mesh_path)
        os.remove(f'{model_dir}/RIR_G_epoch_{stale_epoch_tag}.pth')
        if RIR_D is not None:
            os.remove(f'{model_dir}/RIR_D_epoch_{stale_epoch_tag}.pth')


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def generate_complementary_filterbank(
    fc,
    fs=16000,
    filter_order=4,
    filter_length=16384,
    power=True,
):
    """Return a zero-phase complementary filterbank via Butterworth prototypes."""

    fc = np.sort(fc)
    assert fc[-1] <= fs / 2

    num_filters = len(fc)
    nbins = filter_length
    signal_z1 = np.zeros(2 * nbins)
    signal_z1[0] = 1
    ir_bands = np.zeros((2 * nbins, num_filters))

    for i in range(num_filters - 1):
        wc = fc[i] / (fs / 2.0)

        b_low, a_low = scipy.signal.butter(filter_order, wc, btype='low')
        b_high, a_high = scipy.signal.butter(filter_order, wc, btype='high')

        ir_bands[:, i] = scipy.signal.lfilter(b_low, a_low, signal_z1)
        signal_z1 = scipy.signal.lfilter(b_high, a_high, signal_z1)

    ir_bands[:, -1] = signal_z1

    if power:
        ir2_bands = np.real(
            np.fft.ifft(np.square(np.abs(np.fft.fft(ir_bands, axis=0))), axis=0)
        )
    else:
        ir2_bands = np.real(
            np.fft.ifft(np.abs(np.abs(np.fft.fft(ir_bands, axis=0))), axis=0)
        )

    ir2_bands = np.concatenate(
        (ir2_bands[nbins:(2 * nbins), :], ir2_bands[0:nbins, :]),
        axis=0,
    )
    return ir2_bands
