"""
@author: andrewkof

Run example:
python3 main.py --epochs=10000 --beta=0.5 --gamma=0.5 --data='gmm8'

Reproduces Hellinger case with gmm8 data target.

Visit "Run examples" section of README for more info about the default values and choices.
"""


from Cumulant_GAN import *
from numpy import genfromtxt
import argparse

# -------------------
# input arguments
# -------------------
parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--epochs', '-e', default=10000, type=int, help='[int] How many generator iterations to train for')
parser.add_argument('--beta', '-b', default=0.0, type=float, help='[float] Cumulant GAN beta parameter')
parser.add_argument('--gamma', '-g', default=0.0, type=float, help='[float] Cumulant GAN gamma parameter')
parser.add_argument('--data', '-d', default='gmm8', choices=['gmm8', 'tmm6', 'swissroll'], type=str, help='[str] Data target')

args = parser.parse_args()
beta = args.beta
gamma = args.gamma
epochs = args.epochs
data_name = args.data

name = get_divergence_name(beta, gamma)
data = get_data_name(data_name)

path_to_data = 'data/' + data
toy_data = genfromtxt(path_to_data, delimiter=',', dtype='float32')


if data == 'swiss_roll_2d_with_labels.csv':
    toy_data = toy_data[:, :-1]

SAMPLE_SIZE = toy_data.shape[0]         # gmm8 and tmm6 data have 50k samples while swissroll data has 10k.
BATCH_SIZE = 10000

train_dataset = tf.data.Dataset.from_tensor_slices(toy_data).shuffle(SAMPLE_SIZE).batch(BATCH_SIZE)

G = Cumulant_GAN_dense(name, data_name, beta, gamma, epochs, BATCH_SIZE)           # Initialize GAN
G.train(train_dataset)                                                             # Train it

# Create animated clips
if data == 'toy_example_gmm8.csv':
    create_gmm8_clip(name)

elif data == 'toy_example_tmm6.csv':
    create_tmm6_clip(name)

else:
    create_swiss_roll_clip(name)
