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
from os.path import exists, isfile
from os import mkdir
import argparse
import itertools
from validation_to_png import plot_validation_set
from train import validate, evaluate

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

# Check machine for GPU
if torch.cuda.is_available():
    print("Using a CUDA compatible device")
    cuda = torch.device("cuda:0")
    cpu = torch.device("cpu")
else:
    print("Using the CPU")
    cuda = torch.device("cpu")
    cpu = torch.device("cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="An automated brightness profile fitter")
    parser.add_argument("gals", help="Textfile containing galaxies in the test set.")
    parser.add_argument("checkpoint", help="Model file dict to load.")
    parser.add_argument("--encoding_len", dest="encoding_len", default=512, type=int,
                        help="Length of encoding of images, and of hidden layer in GRU units.")
    args = parser.parse_args()
    print(args)

    # Load checkpoint data
    checkpoint = torch.load(args.checkpoint, map_location=cuda)
    chk_epoch = checkpoint["epoch"]
    chkdir = checkpoint["logdir"]

    # Create log directory
    logdir = "{}/test_set/".format(chkdir)
    print("LOGDIR", logdir)
    if not exists(logdir):
        mkdir(logdir)

    # Generate test set file list and save as textfile
    f_lst = np.genfromtxt(args.gals, dtype=str)
    test_set = f_lst
    np.savetxt(logdir + "test_set.txt",
               sorted([fi for fi in test_set]), fmt="%s")

    # Ensure that test set file list is complete
    chs = ("g", "r", "i")
    test_set = [(fi, ch) for fi, ch in zip(np.repeat(test_set, len(chs)), itertools.cycle(chs)) 
                 if isfile("./data/gals/{}-{}.fits".format(fi, ch))
                 and isfile("./data/sbs_gri_noise/{}_{}.txt".format(fi, ch))]

    # Init and load Pix2Prof
    encoder = ResNet18(num_classes=args.encoding_len).to(cuda)
    decoder = GRUNet(input_dim=1, hidden_dim=args.encoding_len, output_dim=1, n_layers=3).to(cuda)
    criterion = nn.MSELoss()
    encoder_op = optim.Adam(encoder.parameters(), lr=0.0002)
    decoder_op = optim.Adam(decoder.parameters(), lr=0.0002)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    decoder_op.load_state_dict(checkpoint["decoder_op"])
    encoder_op.load_state_dict(checkpoint["encoder_op"])

    # Validate
    validate(test_set, encoder, decoder, chk_epoch, criterion, logdir=logdir)
    plot_validation_set("{}/{:04d}".format(logdir, chk_epoch))
