import torch 
import copy 
from tqdm.auto import tqdm
from models import Model
from loss import contrastive_loss
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt

model_params = {
    'decoder': {
        'latent_dim' : 128,
        'channel_dim' : 128
    },

    'encoder' : {
        'ch_in' : 3,
        'hid_ch': 128,
        'z_dim' : 128
    },

    'discriminator' : {
        'ch_in' : 3, 
        'hid_ch': 128,
        'cont_dim' : 64
    }
}

hparams = {
    'epochs' : 1,
    'train_batch_size' : 10, 
    'test_batch_size' : 10,
    'lr' : 0.01,
    'device' : 'cuda' if torch.cuda.is_available() else 'cpu'
}


def show_img(img : torch.tensor, num_images=25, size=(3, 32, 32)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in a uniform grid.
    '''
    image_unflat = img.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


def train(model_params, hparams):
    
    device = hparams['device']
    model = Model(model_params).to(device)

    enc_optim = torch.optim.Adam(model.encoder.parameters(), lr = hparams['lr'])
    dec_optim = torch.optim.Adam(model.decoder.parameters(), lr = hparams['lr'])
    disc_optim = torch.optim.Adam(model.discriminator.parameters(), lr = hparams['lr'])

    gan_criterion = torch.nn.BCEWithLogitsLoss()

    train_loader = DataLoader(
        torchvision.datasets.CIFAR10('./data', train = True, download = True, transform = torchvision.transforms.ToTensor()),
        batch_size=hparams['train_batch_size'], 
        shuffle=True, 
        drop_last=True
        )

    test_loader = DataLoader(torchvision.datasets.CIFAR10('./data', train = False, transform = torchvision.transforms.ToTensor()),
        batch_size=hparams['test_batch_size'], 
        shuffle=False
        )


    gen_loss_train = []
    disc_loss_train = []
    cont_loss_train = []

    mean_generator_loss = 0
    mean_discriminator_loss = 0
    mean_contrastive_loss = 0

    disp_freq = 500
    step = 1

    iterator = tqdm(range(1,hparams['epochs']+1), leave=True)

    for epoch in iterator:
        
        iterator.set_description_str(f"Epoch: {epoch}")

        for point_batch, _ in train_loader: 
            
            model.train()

            #### Real Data
            real_data = point_batch.to(device) 

            #### Fake Data
            # fake_data = model.gen_from_noise(size=(real_data.size(0), model_params['decoder']['latent_dim']))

            #### Reconstructed Data
            z_latent, rec_data = model(real_data)

            '''----------------         Discriminator Update         ----------------'''
            disc_optim.zero_grad()

            # disc_fake_pred, _ = model.discriminator(fake_data.detach())
            # disc_fake_loss = gan_criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))

            disc_rec_pred, _ = model.discriminator(rec_data.detach())
            disc_rec_loss = gan_criterion(disc_rec_pred, torch.zeros_like(disc_rec_pred))

            disc_real_pred, _ = model.discriminator(real_data)
            disc_real_loss = gan_criterion(disc_real_pred, torch.ones_like(disc_real_pred))

            gan_objective = disc_real_loss + disc_rec_loss # + disc_fake_loss 
            gan_objective.backward()
            disc_optim.step()

            # Log
            disc_loss_train.append(gan_objective.item())
            mean_discriminator_loss += gan_objective.item() / disp_freq

            '''----------------         Generator Update         ----------------'''

            # KLD loss term missing !!!!!
            enc_optim.zero_grad()
            dec_optim.zero_grad()
            
            # fake_data = model.gen_from_noise(size=(real_data.size(0), model_params['decoder']['latent_dim'])).to(device)

            # disc_fake_pred, _ = discriminator(fake_data)
            # gen_fake_loss = gan_criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))

            disc_rec_pred, _ = model.discriminator(rec_data)
            gen_rec_loss = gan_criterion(disc_rec_pred, torch.ones_like(disc_rec_pred))

            gan_objective =  gen_rec_loss # + gen_fake_loss 

            gan_objective.backward()
            enc_optim.step()
            dec_optim.step()

            # Log
            gen_loss_train.append(gan_objective.item())
            mean_generator_loss += gan_objective.item() / disp_freq

            '''----------------         Contrastive Update         ----------------'''

            enc_optim.zero_grad()
            dec_optim.zero_grad()
            disc_optim.zero_grad()

            _, rec_contrastive = model.discriminator(rec_data)
            _, real_contrastive = model.discriminator(real_data)

            cont_loss = contrastive_loss(z_latent, real_contrastive, rec_contrastive)
            
            cont_loss.backward()
            disc_optim.step()
            enc_optim.step()
            dec_optim.step()

            # Log
            cont_loss_train.append(cont_loss.item())
            mean_contrastive_loss += cont_loss.item() / disp_freq
            

            # Visualize the generated images
            if step % disp_freq == 0:
                gen_images = model.gen_from_noise(size = (25, model_params['decoder']['latent_dim']))
                show_img(gen_images, num_images=25, size=(3, 32, 32))
                mean_contrastive_loss = 0
                mean_generator_loss = 0
                mean_discriminator_loss = 0
            
            step += 1

            iterator.set_postfix_str(
                f"Disc Loss: {disc_loss_train[-1]:.4f}, Gen Loss: {gen_loss_train[-1]:.4f}, Cont Loss: {cont_loss_train[-1]:.4f}"
                )


train(model_params, hparams)
