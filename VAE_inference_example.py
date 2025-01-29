import torch # tested on version 2.1.2+cu118
import scipy.io as io
import argparse
import logging
from utils import load_dataset_test, save_image_mat
from fMRIVAE_Model import BetaVAE
import os

def main():

    parser = argparse.ArgumentParser(description='VAE for fMRI generation')
    parser.add_argument('--batch-size', type=int, metavar='N', help='how many samples per saved file?')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--zdim', type=int, default=256, metavar='N', help='dimension of latent variables')
    parser.add_argument('--data-path', type=str, metavar='DIR', help='path to dataset')
    parser.add_argument('--z-path', type=str, default='./result/latent/', help='path to saved z files')
    parser.add_argument('--resume', type=str, default='./checkpoint/checkpoint.pth.tar', help='the VAE checkpoint') 
    parser.add_argument('--img-path', type=str, default='./result/recon', help='path to save reconstructed images')
    parser.add_argument('--mode', type=str, default='both', help='choose from \'encode\',\'decode\' or \'both\'')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode for detailed logging')

    args = parser.parse_args()

    if not os.path.isdir(args.z_path):
        os.system('mkdir '+ args.z_path + ' -p')
    if (args.mode != 'encode') and not os.path.isdir(args.img_path):
        os.system('mkdir '+ args.img_path + ' -p')

    # Set logging level based on debug flag
    logging_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=logging_level, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.debug("Starting the VAE inference script.")
    args = parser.parse_args()
    logging.debug(f"Parsed arguments: {args}")

    try:
        torch.manual_seed(args.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.debug(f"Using device: {device}")

        logging.debug(f"Loading VAE model from {args.resume}.")
        model = BetaVAE(z_dim=args.zdim, nc=1).to(device)
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['state_dict'])
            logging.debug("Checkpoint loaded.")
        else:
            logging.error(f"Checkpoint not found at {args.resume}")
            raise RuntimeError("Checkpoint not found.")

        if (args.mode == 'encode') or (args.mode == 'both'):
            logging.debug("Starting encoding process...")
            test_loader = load_dataset_test(args.data_path, args.batch_size)
            logging.debug(f"Loaded test dataset from {args.data_path}")
            for batch_idx, (xL, xR) in enumerate(test_loader):
                xL = xL.to(device)
                xR = xR.to(device)
                z_distribution = model._encode(xL, xR)
                save_data = {'z_distribution': z_distribution.detach().cpu().numpy()}
                io.savemat(os.path.join(args.z_path, f'save_z{batch_idx}.mat'), save_data)
                logging.debug(f"Encoded batch {batch_idx}")

        if (args.mode == 'decode') or (args.mode == 'both'):
            logging.debug("Starting decoding process...")
            filelist = [f for f in os.listdir(args.z_path) if f.split('_')[0] == 'save']
            logging.debug(f"Filelist: {filelist}")
            for batch_idx, filename in enumerate(filelist):
                logging.debug(f"Decoding file {filename}")
                z_dist = io.loadmat(os.path.join(args.z_path, f'save_z{batch_idx}.mat'))
                z_dist = z_dist['z_distribution']
                mu = z_dist[:, :args.zdim]
                z = torch.tensor(mu).to(device)
                x_recon_L, x_recon_R = model._decode(z)
                save_image_mat(x_recon_R, x_recon_L, args.img_path, batch_idx)
                logging.debug(f"Decoded and saved batch {batch_idx}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()
