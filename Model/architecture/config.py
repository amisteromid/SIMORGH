#config_model = {
#    "dims": {'_MAX_RESIDUE_TYPE': 24, '_NODE_STATE_IRREPS': '128x0e+64x1e+32x2e+16x3e',
#    '_NODE_FEATURES_IRREPS': '1x1e+128x0e', '_NUM_RADIAL': 128,
#    '_EDGE_ATTR_IRREPS': '1x1e+1x1e', '_IRREPS_HEAD': '16x0e+8x1e+4x2e+2x3e', '_NUM_HEADS': 4},
#    "layers": [11,11,11,11],
#    }

config_model = {
    "dims": {'_MAX_RESIDUE_TYPE': 24, '_NODE_STATE_IRREPS': '128x0e+64x1e+32x2e',
    '_NODE_FEATURES_IRREPS': '1x1e+1x2e+128x0e', '_NUM_RADIAL': 128,
    '_EDGE_ATTR_IRREPS': '1x1e+1x2e+1x1e+1x2e', '_IRREPS_HEAD': '16x0e+8x1e+4x2e', '_NUM_HEADS': 4},
    "layers": [11,11,11,11],
    }

config_runtime = {'loss_alpha': 0.75,
    'loss_gamma': 2,
    'patience': 40,
    'log_step': 512, 
    'device': 'cuda',
    'batch_size': 2,
    'num_epochs': 50,
    'min_lr': 5e-6,
    'warmup_epochs': 2,
    'max_lr': 8e-4,
    }


config_runtime_set = {'loss_alpha': 0.75,
    'loss_gamma': 2,
    'patience': 25,
    'log_step': 64, 
    'device': 'cuda',
    'batch_size': 1,
    'num_epochs': 20, 
    'min_lr': 1e-6,
    'warmup_epochs': 2,
    'max_lr': 5e-5,
    }

#config_data = {
#    'dataset_filepath': "/home/omokhtari/plbs/Data/ds3/db_bbflow.h5",
#    'train_selection_filepath': '/home/omokhtari/plbs/Data/splitting/train_p2rank.txt',
#    'valid_selection_filepath': '/home/omokhtari/plbs/Data/splitting/valid_p2rank.txt',
#    'max_size': 1500,
#}

#config_data = {
#    'dataset_filepath': "/home/omokhtari/plbs/Data/ds3/db_bbflow.h5",
#    'train_selection_filepath': '/home/omokhtari/plbs/Data/ds3/splitting/train.txt',
#    'valid_selection_filepath': '/home/omokhtari/plbs/Data/ds3/splitting/valid.txt',
#    'max_size': 1500,
#}
#config_data = {
#    'dataset_filepath': "/home/omokhtari/plbs/Data/AFlow_raw/db_AFlow_raw.h5",
#    'train_selection_filepath': '/home/omokhtari/plbs/Data/AFlow_raw/splitting/train.txt',
#    'valid_selection_filepath': '/home/omokhtari/plbs/Data/AFlow_raw/splitting/valid.txt',
#    'max_size': 1500,
#}
#config_data = {
#    'dataset_filepath': "/home/omokhtari/plbs/Data/data/db4.h5",
#    'train_selection_filepath': '/home/omokhtari/plbs/Data/Misato/splitting/train3.txt',
#    'valid_selection_filepath': '/home/omokhtari/plbs/Data/Misato/splitting/valid3.txt',
#    'max_size': 1500,
#}

#config_data = {
#    'dataset_filepath': "/home/omokhtari/plbs/Data/data/db5.h5",
#    'train_selection_filepath': '/home/omokhtari/plbs/Data/cryptobench/splitting/train.txt',
#    'valid_selection_filepath': '/home/omokhtari/plbs/Data/cryptobench/splitting/valid.txt',
#    'max_size': 1500,
#}

#config_data = {
#    'dataset_filepath': "/srv/storage/delta@storage4.nancy.grid5000.fr/omokhtari/db_plinder.h5",
#    'train_selection_filepath': '/home/omokhtari/SIMORGH/Data/plinder//train_manual.txt',
#    'valid_selection_filepath': '/home/omokhtari/SIMORGH/Data/plinder/valid_manual.txt',
#    'max_size': 2000,
#}

config_data = {
    'dataset_filepath': "/srv/storage/delta@storage4.nancy.grid5000.fr/omokhtari/db5.h5",
    'train_selection_filepath': '/home/omokhtari/SIMORGH/Data/asd/split/train.txt',
    'valid_selection_filepath': '/home/omokhtari/SIMORGH/Data/asd/split/valid.txt',
    'max_size': 2000,
}
