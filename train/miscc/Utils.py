import errno
import os
import sys
from collections import OrderedDict
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train.Model.MESH_encoder.MESH_model import MESH_encoder
from train.Model.RIR_decoder.RIR_Dec_model import RIR_decoder


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def _strip_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_state_dict[key.replace('module.', '')] = value
    return new_state_dict


def load_model_train(model_cfg):
    model_dir = model_cfg.DATA.out_path
    mkdir_p(model_dir)
    mkdir_p(os.path.join(model_dir, 'Model'))

    mesh_enc = MESH_encoder(model_cfg)
    with open(os.path.join(model_dir, 'Mesh_Enc.txt'), 'w', encoding='utf-8') as f:
        print(mesh_enc, file=f)

    rir_g = RIR_decoder(model_cfg)
    with open(os.path.join(model_dir, 'RIR_G.txt'), 'w', encoding='utf-8') as f:
        print(rir_g, file=f)

    if model_cfg.TRAIN.MESH_NET != '':
        mesh_path = os.path.join(model_dir, 'Model', model_cfg.TRAIN.MESH_NET)
        state_dict = torch.load(mesh_path, map_location=lambda storage, loc: storage)
        mesh_enc.load_state_dict(_strip_module_prefix(state_dict))
        print('Load from:', mesh_path)

    if model_cfg.TRAIN.NET_G != '':
        net_g_path = os.path.join(model_dir, 'Model', model_cfg.TRAIN.NET_G)
        state_dict = torch.load(net_g_path, map_location=lambda storage, loc: storage)
        rir_g.load_state_dict(state_dict)
        print('Load from:', net_g_path)

    return mesh_enc, rir_g, None


def save_model(mesh_enc, rir_g, rir_d, epoch, model_dir):
    if mesh_enc is not None:
        torch.save(mesh_enc.state_dict(), '%s/Mesh_Enc_epoch_%s.pth' % (model_dir, str(epoch)))
    if rir_g is not None:
        torch.save(rir_g.state_dict(), '%s/RIR_G_epoch_%s.pth' % (model_dir, str(epoch)))


def save_newest_model(mesh_enc, rir_g, rir_d, model_dir, epoch):
    if mesh_enc is not None:
        torch.save(mesh_enc.state_dict(), '%s/Mesh_Enc_epoch_%s.pth' % (model_dir, str(epoch).zfill(4)))
    if rir_g is not None:
        torch.save(rir_g.state_dict(), '%s/RIR_G_epoch_%s.pth' % (model_dir, str(epoch).zfill(4)))

    old_mesh_path = '%s/Mesh_Enc_epoch_%s.pth' % (model_dir, str(epoch - 4).zfill(4))
    old_rir_path = '%s/RIR_G_epoch_%s.pth' % (model_dir, str(epoch - 4).zfill(4))
    if os.path.exists(old_mesh_path):
        os.remove(old_mesh_path)
    if os.path.exists(old_rir_path):
        os.remove(old_rir_path)



def load_model_test(model_cfg):
    """Load the fixed main model for inference from cfg.TEST checkpoints."""
    model_dir = model_cfg.DATA.out_path
    mesh_enc = MESH_encoder(model_cfg)
    rir_g = RIR_decoder(model_cfg)

    mesh_name = getattr(getattr(model_cfg, 'TEST', object()), 'MESH_NET', 'Mesh_Enc_epoch_best.pth')
    rir_name = getattr(getattr(model_cfg, 'TEST', object()), 'NET_G', 'RIR_G_epoch_best.pth')
    mesh_path = os.path.join(model_dir, 'Model', mesh_name)
    rir_path = os.path.join(model_dir, 'Model', rir_name)

    if os.path.exists(mesh_path):
        state_dict = torch.load(mesh_path, map_location=lambda storage, loc: storage)
        mesh_enc.load_state_dict(_strip_module_prefix(state_dict))
        print('Load from:', mesh_path)
    else:
        raise FileNotFoundError(mesh_path)

    if os.path.exists(rir_path):
        state_dict = torch.load(rir_path, map_location=lambda storage, loc: storage)
        rir_g.load_state_dict(_strip_module_prefix(state_dict))
        print('Load from:', rir_path)
    else:
        raise FileNotFoundError(rir_path)

    return mesh_enc, rir_g
