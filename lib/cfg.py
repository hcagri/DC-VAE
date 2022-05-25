import torch

configs = {
    'model_params' : {
        'decoder': {
            'latent_dim' : 128,
            'channel_dim' : 256
        },

        'encoder' : {
            'ch_in' : 3,
            'hid_ch': 128,
            'z_dim' : 128
        },

        'discriminator' : {
            'ch_in' : 3, 
            'hid_ch': 128,
            'cont_dim' : 16
        }
    },

        'hparams' : {
        'epochs' : 800,
        'train_batch_size' : 64, 
        'test_batch_size' : 64,
        'lr' : 0.0002,
        'disp_freq' : 20,
        'gen_train_freq' : 5,
        'checkpoint': 500,
        'device' : torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }
}