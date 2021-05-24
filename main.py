from torch.utils.data import DataLoader

from attention import AttnDecoderRNN
from dataset import ProductDescriptionDataset, Vocabulary
from encoder import EncoderRNN
from train import trainIters
from variables import *

def load_datasets():
    train_dataset = ProductDescriptionDataset(data_path='./data/train.csv')
    validation_dataset = ProductDescriptionDataset(data_path='./data/val.csv')
    train_tokens = train_dataset.get_tokens_list()
    validation_tokens = validation_dataset.get_tokens_list()
    tokens = set(train_tokens + validation_tokens)
    vocab = Vocabulary(sorted(tokens), add_unk_token=True)
    train_dataset.set_vocab(vocab)
    validation_dataset.set_vocab(vocab)
    return train_dataset, validation_dataset

if __name__ == '__main__':
    # hidden_size = 256
    # encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    # attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
    #
    # trainIters(encoder1, attn_decoder1, 75000, print_every=5000)

    train_dataset, validation_dataset = load_datasets()

    batch_size = 32
    train_dataloader = DataLoader(train_dataset, batch_size)
    validation_dataloader = DataLoader(validation_dataset, batch_size)
