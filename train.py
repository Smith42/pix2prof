"""
CNN stem to RNN classifier for brightness profile fitting

The idea here is that we can model a brightness profile fitting function 
similarly to the way that we model a image to natural language caption 
function (i.e. Image -> CNN encoder -> RNN decoder -> Caption/Data)

@AUTHOR Mike Smith December 2019
@EMAIL mikejamesjsmith <at> gmail <dot> com
"""
# Other imports
import numpy as np
import glob
from tqdm import trange
from os.path import exists, isfile
from os import mkdir
from astropy.io import fits
import random
import argparse
import time
import itertools
from validation_to_png import plot_validation_set
from math import trunc

# ML imports
import torch
import torch.nn as nn
import torch.optim as optim
from resnet import ResNet18
from grunet import GRUNet

# Global vars
SOS_TOKEN = torch.full((1, 1, 1), 0.0)
SKY = 30.0
MAX_LENGTH = 1024

# Check if GPU is available
if torch.cuda.is_available():
    print("Using a CUDA compatible device")
    cuda = torch.device("cuda:0")
    cpu = torch.device("cpu")
else:
    print("Using the CPU")
    cuda = torch.device("cpu")
    cpu = torch.device("cpu")

def mmn(ar, mini=2.0, maxi=30.0):
    return (ar - mini)/(maxi - mini)

def get_im(fi, width=256):
    """
    Get and augment galaxy image FITS file given a file name.
    Also get the corresponding SB profile.
    """
    ch = fi[1]
    fi = fi[0]
    
    galaxy_fi = "./data/gals/{}-{}.fits".format(fi, ch)
    profile_fi = "./data/sbs_gri_noise/{}_{}.txt".format(fi, ch)
    #profile_fi = "./data/sbs_gri_30.0/{}_{}.txt".format(fi, ch)

    with fits.open(galaxy_fi) as hdul:
        galaxy = hdul[0].data

    galaxy[galaxy >= np.percentile(galaxy, 99.9)] = np.percentile(galaxy, 99.9)
    galaxy = mmn(galaxy)

    # we add wobble to encourage the encoder to be galaxy position agnostic
    #wobble = (np.random.randint(-50, 50), np.random.randint(-50, 50))
    wobble = tuple(map(trunc, 10 * np.random.randn(2)))

    mp = (galaxy.shape[0]//2, galaxy.shape[0]//2)
    galaxy = galaxy[np.newaxis, mp[0] - (width//2) + wobble[0]:mp[0] + (width//2) + wobble[0],
                    mp[1] - (width//2) + wobble[1]:mp[1] + (width//2) + wobble[1]]
    # we need to convert `galaxy` to a little endian dtype for pytorch compatability....
    galaxy = galaxy.astype(np.float32)
    # A random rotation:
    galaxy = np.rot90(galaxy, k=np.random.randint(0, 4), axes=[1, 2])

    profile = np.loadtxt(profile_fi)

    return galaxy, profile 

def im_generator(f_lst, batch_size=1):
    """
    Image generator for getting galaxy -- surface brightness profile pairs.
    """
    while True:
         width = 256 #random.choice([256, 512, 1024])
         galaxy, profile = zip(*[get_im(random.choice(f_lst), width) for _ in range(batch_size)])
         galaxy = torch.Tensor(galaxy).to(cuda)
         profile = torch.Tensor(profile).to(cuda)
         
         yield galaxy, profile

def train(galaxy, gt_profile, encoder, decoder, encoder_op, decoder_op, criterion):
    encoder_op.zero_grad()
    decoder_op.zero_grad()

    profile_length = gt_profile.size(1)

    # Encode the galaxy image to z (== h0)
    galaxy_enc = encoder(galaxy)

    loss = 0

    decoder_input = SOS_TOKEN.to(cuda)
    # This is needed to get the data into the correct shape for the GRU network
    # 3 layers
    decoder_hidden = galaxy_enc.repeat(3, 1, 1)

    # Teacher force at a rate defined by teacher_forcing_ratio
    use_teacher_forcing = True if random.random() < args.teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing will feed the GT as the next input
        for di in range(profile_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss = loss + criterion(decoder_output.squeeze(), gt_profile[:, di].squeeze())
            decoder_input = gt_profile[:, di:di+1].unsqueeze(0)

    else:
        profile = []
        # No teacher forcing; use decoder output as input until sky magnitude (EOS threshold) reached (27.5)
        for di in range(profile_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss = loss + criterion(decoder_output.squeeze(), gt_profile[:, di].squeeze())

            decoder_input = decoder_output.detach().unsqueeze(0)
            profile.append(decoder_output.item())

            # If we hit background sky break the loop
            if len(profile) > 100 and np.std(profile[-100:-1]) <= 0.01:
                break

    loss.backward()

    encoder_op.step()
    decoder_op.step()

    return loss.item() / profile_length

def evaluate(encoder, decoder, galaxy, max_length=MAX_LENGTH):
    with torch.no_grad():
        galaxy_enc = encoder(galaxy)

        decoder_input = SOS_TOKEN.to(cuda)
        decoder_hidden = galaxy_enc.repeat(3, 1, 1)

        profile = []

        for di in range(MAX_LENGTH):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)

            # If we hit background sky break the loop
            if len(profile) > 100 and np.std(profile[-100:-1]) <= 0.01:
                profile.append(decoder_output.item())
                break
            else:
                profile.append(decoder_output.item())

            decoder_input = decoder_output.detach().unsqueeze(0)

        return torch.Tensor(profile)

def validate(f_lst, encoder, decoder, epoch, criterion, logdir=None):
    losses = []
    p_lst = []
    y_lst = []

    for b in trange(len(f_lst)):
        fi = f_lst[b]
        galaxy, profile = next(im_generator([fi], batch_size=1))
        profile = profile.cpu().squeeze()

        p_profile = evaluate(encoder, decoder, galaxy)

        y_lst.append(profile.cpu().numpy())
        p_lst.append(p_profile.numpy())
        min_length = list(min((profile.squeeze().shape, p_profile.shape)))[0]
        loss = criterion(p_profile[:min_length], profile[:min_length])
        losses.append(loss.item())

    print("Validation:", np.mean(losses), np.std(losses))

    if logdir is not None:
        epochdir = "{}/{:04d}/".format(logdir, epoch)
        if not exists(epochdir):
            mkdir(epochdir)

        for fi, y, p in zip(f_lst, y_lst, p_lst):
            np.savetxt("{}{}-{}-y.txt".format(epochdir, fi[0], fi[1]), y)
            np.savetxt("{}{}-{}-p.txt".format(epochdir, fi[0], fi[1]), p)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="An automated brightness profile fitter")
    parser.add_argument("--seed", dest="seed", default=42, help="A random seed for reproducability")
    parser.add_argument("--teacher_forcing_ratio", dest="teacher_forcing_ratio", 
                        default=0.4, help="Probability of using teacher forcing")
    parser.add_argument("--gals", dest="gals", default="./filtered_intersection.txt", 
                        help="Textfile containing galaxies in training + validation set.")
    parser.add_argument("--encoding_len", dest="encoding_len", default=512, type=int,
                        help="Length of encoding of images, and of hidden layer in GRU units.")
    parser.add_argument("--logdir", dest="logdir", help="Log directory")
    parser.add_argument("--epochs", dest="epochs", default=200, type=int, help="Number of epochs")
    parser.add_argument("--checkpoint", dest="checkpoint", help="Model file dict to load.")
    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    chk_epoch = 0
    batches = 500
    batch_size = 1
    valid_size = 64
    test_size = 100

    # Create and print logdir
    if args.logdir is None:
        logdir = "logs/{}/".format(int(time.time()))
    else:
        logdir = args.logdir
    print("LOGDIR", logdir)
    if not exists(logdir):
        mkdir(logdir)

    # Generate and save training, test, and validation set galaxy file lists
    f_lst = np.random.permutation(np.genfromtxt(args.gals, dtype=str))
    valid_set = f_lst[:valid_size]
    test_set = f_lst[valid_size:valid_size + test_size]
    training_set = f_lst[valid_size + test_size:]

    np.savetxt(logdir + "training_set.txt",
               sorted([fi for fi in training_set]), fmt="%s")
    np.savetxt(logdir + "valid_set.txt",
               sorted([fi for fi in valid_set]), fmt="%s")
    np.savetxt(logdir + "test_set.txt",
               sorted([fi for fi in test_set]), fmt="%s")

    # Make sure file lists are complete
    chs = ("g", "r", "i")
    valid_set = [(fi, ch) for fi, ch in zip(np.repeat(valid_set, len(chs)), itertools.cycle(chs)) 
                 if isfile("./data/gals/{}-{}.fits".format(fi, ch))
                 and isfile("./data/sbs_gri_30.0/{}_{}.txt".format(fi, ch))]
    training_set = [(fi, ch) for fi, ch in zip(np.repeat(training_set, len(chs)), itertools.cycle(chs)) 
                    if isfile("./data/gals/{}-{}.fits".format(fi, ch))
                    and isfile("./data/sbs_gri_30.0/{}_{}.txt".format(fi, ch))]
                    

    # Init Pix2Prof and load checkpoint if asked
    encoder = ResNet18(num_classes=args.encoding_len).to(cuda)
    decoder = GRUNet(input_dim=1, hidden_dim=args.encoding_len, output_dim=1, n_layers=3).to(cuda)
    criterion = nn.MSELoss()
    encoder_op = optim.Adam(encoder.parameters(), lr=0.0002)
    decoder_op = optim.Adam(decoder.parameters(), lr=0.0002)

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        encoder.load_state_dict(checkpoint["encoder"])
        decoder.load_state_dict(checkpoint["decoder"])
        decoder_op.load_state_dict(checkpoint["decoder_op"])
        encoder_op.load_state_dict(checkpoint["encoder_op"])
        chk_epoch = checkpoint["epoch"]
        logdir = checkpoint["logdir"]

    # Train for `args.epochs` number of epochs
    for epoch in range(chk_epoch, chk_epoch + args.epochs + 1):
        running_loss = 0.0

        # Validate the validation set...
        validate(valid_set, encoder, decoder, epoch, criterion, logdir=logdir)
        epochdir = "{}/{:04d}/".format(logdir, epoch)
        plot_validation_set(epochdir)

        if epoch % 10 == 0:
            # Checkpoint every 10 epochs
            torch.save({"encoder": encoder.state_dict(),
                        "decoder": decoder.state_dict(),
                        "encoder_op": encoder_op.state_dict(),
                        "decoder_op": decoder_op.state_dict(),
                        "epoch": epoch,
                        "logdir": logdir},
                        "{}/checkpoint.pth".format(epochdir))

        print("Epoch", epoch)

        # Train!
        with trange(batches) as bs:
            for b in bs:
                galaxy, profile = next(im_generator(training_set, batch_size=1))
                galaxy, profile = galaxy.to(cuda), profile.to(cuda) 

                loss = train(galaxy, profile, encoder, decoder, encoder_op, decoder_op, criterion)

                # Metrics
                running_loss = running_loss + loss
                bs.set_description("{:.4f}".format(loss))

        print(running_loss / batches)
