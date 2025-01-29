import h5py
import torch
import torch.utils.data as data
import torch.multiprocessing
import scipy.io as sio
from torch.nn import functional as F
# torch.multiprocessing.set_start_method('spawn')

class H5Dataset(data.Dataset):
    def __init__(self, H5Path):
        super(H5Dataset, self).__init__()
        self.H5File = h5py.File(H5Path,'r')       
        self.LeftData = self.H5File['LeftData']
        self.RightData = self.H5File['RightData']
        #self.LeftMask = self.H5File['LeftMask'][:] # update 2024.01.11 Masks loaded separately
        #self.RightMask = self.H5File['RightMask'][:]

    def __getitem__(self, index):
        return (torch.from_numpy(self.LeftData[index,:,:,:]).float(),
                torch.from_numpy(self.RightData[index,:,:,:]).float())
 
    def __len__(self):
        return self.LeftData.shape[0]

def save_image_mat(img_r, img_l, result_path, idx):
    save_data = {}
    save_data['recon_L'] = img_l.detach().cpu().numpy()
    save_data['recon_R'] = img_r.detach().cpu().numpy()
    sio.savemat(result_path+'img{}.mat'.format(idx), save_data)

def load_dataset(data_path, batch_size):
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_dir = data_path + '_train.h5'
    val_dir = data_path + '_val.h5'
    train_set = H5Dataset(train_dir)
    val_set = H5Dataset(val_dir)
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size, shuffle=False, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_set,batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, val_loader

def load_dataset_test(data_path, batch_size):
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    test_dir = data_path + '.h5'
    test_set = H5Dataset(test_dir)
    test_loader = torch.utils.data.DataLoader(test_set,batch_size=batch_size, shuffle=False, **kwargs)
    return test_loader

# loss function # update 20240109 mask out zeros
def loss_function(xL, xR, x_recon_L, x_recon_R, mu, logvar, beta, left_mask, right_mask):

    Image_Size=xL.size(3)

    beta/=Image_Size**2
    
    # print('====> Image_Size: {} Beta: {:.8f}'.format(Image_Size, beta))

    # R_batch_size=xR.size(0)
    # Tutorial on VAE Page-14
    # log[P(X|z)] = C - \frac{1}{2} ||X-f(z)||^2 // \sigma^2 
    #             = C - \frac{1}{2} \sum_{i=1}^{N} ||X^{(i)}-f(z^{(i)}||^2 // \sigma^2
    #             = C - \farc{1}{2} N * F.mse_loss(Xhat-Xtrue) // \sigma^2
    # log[P(X|z)]-C = - \frac{1}{2}*2*192*192//\sigma^2 * F.mse_loss
    # Therefore, vae_beta = \frac{1}{36864//\sigma^2}
        
    # mask out zeros
    valid_mask_L = xL!=0
    valid_mask_R = xR!=0 

    if left_mask is not None:
        valid_mask_L = valid_mask_L & (left_mask.detach().to(torch.int32)==1)
        valid_mask_R = valid_mask_R & (right_mask.detach().to(torch.int32)==1)
    
    MSE_L = F.mse_loss(x_recon_L*valid_mask_L, xL*valid_mask_L, size_average=True)
    MSE_R = F.mse_loss(x_recon_R*valid_mask_R, xR *valid_mask_R, size_average=True)  

    # KLD is averaged across batch-samples
    KLD = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1).mean()

    return KLD * beta + MSE_L + MSE_R

