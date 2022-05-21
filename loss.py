import torch 
from torch.nn import CosineSimilarity

def loss(logvar, mu, x, f_z, expectation = True):
    ''' L_instance + KL 

    Args:
        logvar  : output of encoder network, variance, [batch_size, latent_size]
        mu      : output of the encoder network, mean, [batch_size, latent_size]
        x       : original inputs, [batch_size, image_size]
        f_z     : reconstructed images, [batch_size, image_size]
    
    Return:
        contrastive loss

    '''

    KLD = 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2), dim=1)

    distances = h_cossim(x, f_z).exp()
    positive_samples = torch.diag(distances) # diagonal elements of dist are positive pairs.
    negative_samples = torch.sum(distances, dim=0) # sum of columns gives the union of positive and negative samples

    l_instance = -torch.log(torch.div(positive_samples, negative_samples))

    if expectation:
        l_instance = torch.mean(l_instance)
        KLD = KLD.mean()
    
    return l_instance + KLD


def h_cossim(x, y, row_wise = False):
    ''' Calculate Cosine Similarity
    '''

    x = x.unsqueeze(0) if x.dim() == 1 else x 
    y = y.unsqueeze(0) if y.dim() == 1 else y

    # if row_wise:
    #    cos = CosineSimilarity(dim=1)
    #    return cos(x,y)

    num = torch.matmul(x, y.t())
    denum = torch.max(torch.outer(torch.linalg.norm(x, dim=1), torch.linalg.norm(y, dim=1)), torch.tensor(1e-8))

    return num/denum



