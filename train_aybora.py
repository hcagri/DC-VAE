import torch 
from tqdm.auto import tqdm
from lib.models import Model
from loss import contrastive_loss
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import os

# torch.autograd.set_detect_anomaly(True)

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
    'device' : torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

def weights_init(m):
    init_type="xavier_uniform"
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_uniform(m.weight.data, 1.)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)



def show_img(img : torch.tensor, num_images=25, size=(3, 32, 32)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in a uniform grid.
    '''
    image_unflat = img.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    #image_grid = np.uint8((255*image_grid.numpy()))
    image_grid = image_grid.permute(1, 2, 0).squeeze().numpy()
    image_grid = np.uint8(127.5 * image_grid + 127.5)
    plt.imshow(image_grid)
    plt.show()

#taken from orig implementation

def denorm(img, mean, std):
    img = img.clone().detach()
    # img shape is B, 3,64,64 and detached
    for i in range(3):
        img[:, i,:,:] *= std[i]
        img[:, i,:,:] += mean[i]
    return img

def disp_images(img, fname, nrow, norm="none"):
    if not os.path.exists(os.path.dirname(fname)):
        os.makedirs(os.path.dirname(fname))
    bs = img.shape[0]
    imsize = img.shape[2]
    nc = img.shape[1]
    if nc==3 and norm=="0.5":
        img = denorm(img,mean=[0.5, 0.5, 0.5], std=[0.5,0.5,0.5])
    elif nc==3 and norm=="none":
        pass
    elif nc==1:
        img = img
    else:
        raise ValueError("image has incorrect channels")
    img = img.view(bs,-1,imsize,imsize).cpu()
    grid =  torchvision.utils.make_grid(img,nrow=nrow)
    torchvision.utils.save_image(grid, fname)


def train(model_params, hparams):
    
    device = hparams['device']
    model = Model(model_params).to(device)

    # model.apply(weights_init)

    enc_optim = torch.optim.Adam(model.encoder.parameters(), lr = hparams['lr'])
    dec_optim = torch.optim.Adam(model.decoder.parameters(), lr = hparams['lr'])
    disc_optim = torch.optim.Adam(model.discriminator.parameters(), lr = hparams['lr'])

    shared_params = list(model.discriminator.block1.parameters()) + \
                list(model.discriminator.block2.parameters()) + \
                list(model.discriminator.block3.parameters()) + \
                list(model.discriminator.block4.parameters()) + \
                list(model.discriminator.lin.parameters())
    opt_shared = torch.optim.Adam(shared_params,
                                hparams['lr'])
    opt_disc_head = torch.optim.Adam(model.discriminator.disc_lin.parameters(),
                hparams['lr'])
    cont_params = list(model.discriminator.cont_conv.parameters()) + \
                list(model.discriminator.cont_lin.parameters())
    opt_cont_head = torch.optim.Adam(cont_params,
                    hparams['lr'])

    gan_criterion = torch.nn.BCELoss()

    train_loader = DataLoader(
        torchvision.datasets.CIFAR10('./data', train = True, download = True, transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)),])),
        batch_size=hparams['train_batch_size'], 
        shuffle=True, 
        drop_last=True
        )

    test_loader = DataLoader(torchvision.datasets.CIFAR10('./data', train = False, transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)),])),
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
            model.device = device

            #### Real Data
            real_data = point_batch.to(device) 

            #### Fake Data
            fake_data = model.gen_from_noise(size=(real_data.size(0), model_params['decoder']['latent_dim'])).detach()

            #### Reconstructed Data
            z_latent, rec_data = model(real_data)

            '''----------------         Discriminator Update         ----------------'''
            #disc_optim.zero_grad()
            opt_shared.zero_grad()
            opt_disc_head.zero_grad()
            
            # Encoder zero grad

            disc_fake_pred, _ = model.discriminator(fake_data)
            disc_fake_loss = gan_criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))

            disc_rec_pred, _ = model.discriminator(rec_data)
            disc_rec_loss = gan_criterion(disc_rec_pred, torch.zeros_like(disc_rec_pred))

            disc_real_pred, _ = model.discriminator(real_data)
            disc_real_loss = gan_criterion(disc_real_pred, torch.ones_like(disc_real_pred))

            gan_objective = disc_real_loss + disc_rec_loss + disc_fake_loss 
            gan_objective.backward()#retain_graph = True)
            #disc_optim.step()
            opt_shared.step()
            opt_disc_head.step()

            # Log
            disc_loss_train.append(gan_objective.item())
            mean_discriminator_loss += gan_objective.item() / disp_freq

            '''----------------         Generator Update         ----------------'''

            # KLD loss term missing !!!!!
            enc_optim.zero_grad()
            dec_optim.zero_grad()
            
            fake_data = model.gen_from_noise(size=(real_data.size(0), model_params['decoder']['latent_dim']))
            z_latent, rec_data = model(real_data)

            disc_fake_pred, _ = model.discriminator(fake_data)
            gen_fake_loss = gan_criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))

            disc_rec_pred, _ = model.discriminator(rec_data)
            gen_rec_loss = gan_criterion(disc_rec_pred, torch.ones_like(disc_rec_pred))

            gan_objective =  gen_rec_loss + gen_fake_loss 

            gan_objective.backward()#retain_graph = True)
            enc_optim.step()
            dec_optim.step()

            # Log
            gen_loss_train.append(gan_objective.item())
            mean_generator_loss += gan_objective.item() / disp_freq
            
            '''----------------         Contrastive Update         ----------------'''

            enc_optim.zero_grad()
            dec_optim.zero_grad()
            #disc_optim.zero_grad()
            opt_shared.zero_grad()
            opt_cont_head.zero_grad()

            z_latent, rec_data = model(real_data)

            _, rec_contrastive = model.discriminator(rec_data)
            _, real_contrastive = model.discriminator(real_data)


            cont_loss = contrastive_loss(z_latent, real_contrastive, rec_contrastive)
            #print("cont_loss, ", cont_loss)
            
            cont_loss.backward()
            #disc_optim.step()
            opt_shared.step()
            opt_cont_head.step()
            enc_optim.step()
            dec_optim.step()

            # Log
            cont_loss_train.append(cont_loss.item())
            mean_contrastive_loss += cont_loss.item() / disp_freq
            
    
            # Visualize the generated images
            if step % disp_freq == 0:
                # gen_images = model.gen_from_noise(size = (25, model_params['decoder']['latent_dim']))
                # show_img(gen_images, num_images=25, size=(3, 32, 32))
                # mean_contrastive_loss = 0
                # mean_generator_loss = 0
                # mean_discriminator_loss = 0

                viz_img = real_data[0:8].view(8,3,32,32)
                viz_rec = rec_data[0:].view(hparams['train_batch_size'],3,32,32)
                out = torch.cat((viz_img, viz_rec), dim=0)
                fname = os.path.join(f'viz', f"{step}_recon.png")
                disp_images(out, fname, 8, norm="0.5")
                fname = os.path.join(f'viz', f"{step}_sample.png")
                disp_images(fake_data.view(-1,3,32,32), fname, 8, norm="0.5")
            
            step += 1

            iterator.set_postfix_str(
                f"Disc Loss: {disc_loss_train[-1]:.4f}, Gen Loss: {gen_loss_train[-1]:.4f} Cont Loss: {cont_loss_train[-1]:.4f}" # Cont Loss: {cont_loss_train[-1]:.4f}
                )


train(model_params, hparams)
# model = Model(model_params)

# x = torch.randn((10, 3, 32, 32))
# a, b = model.discriminator(x)
# print(a.size())
# print(b.size())