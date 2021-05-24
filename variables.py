import torch
import random
import matplotlib.ticker as ticker

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EOS_TOKEN = 1
SOS_TOKEN = 0
TEACHER_FORCING_RATIO = 0.5

MAX_LENGTH = 700
