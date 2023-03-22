"""Main script to train a model"""
import argparse
import json
from mvrss.utils.functions import count_params
from mvrss.learners.initializer import Initializer
from mvrss.learners.model import Model
from mvrss.models import TMVANet, MVNet, RADnet_2D_downsample, RADnet_2D_downsample2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='Path to config file.',
                        default='config.json')
    args = parser.parse_args()
    cfg_path = args.cfg
    with open(cfg_path, 'r') as fp:
        cfg = json.load(fp)

    init = Initializer(cfg)
    data = init.get_data()
    if 'RAD' in cfg['data_type']:
        if 'mod' in cfg['data_type']:
            if cfg['data_type']=='RAD_mod':      # for 128x128x32 RAD tensor (parallel mode)
                net = RADnet_2D_downsample(n_classes=data['cfg']['nb_classes'],
                                        n_frames=data['cfg']['nb_input_channels'])
            elif cfg['data_type']=='RAD_mod2':   # for 64x64x32 RAD tensor (parallel mode)
                net = RADnet_2D_downsample2(n_classes=data['cfg']['nb_classes'],
                                        n_frames=data['cfg']['nb_input_channels'])
    else:
        if cfg['model'] == 'mvnet':
            net = MVNet(n_classes=data['cfg']['nb_classes'],
                        n_frames=data['cfg']['nb_input_channels'])
        else:
            net = TMVANet(n_classes=data['cfg']['nb_classes'],
                        n_frames=data['cfg']['nb_input_channels'])

    print('Number of trainable parameters in the model: %s' % str(count_params(net)))

    if cfg['model'] == 'mvnet':
        Model(net, data).train(add_temp=False)
    else:
        Model(net, data).train(add_temp=True)

if __name__ == '__main__':
    main()
