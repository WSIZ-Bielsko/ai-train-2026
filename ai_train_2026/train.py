import os
from random import shuffle

import torch
from dotenv import load_dotenv
from loguru import logger

from ai_train_2026.common import get_trainset
from ai_train_2026.tech_gpt_neox import LmAPI, GPTNeoXFNet


def train_on_file(fnet: LmAPI, file_name: str, epochs=3, n_sentences = 10_000):
    logger.info(f"Training on {file_name} for {epochs} epochs")
    trainset: list[str]  = get_trainset(file_name, count = 10**9)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.warning('training on device: ' + device)

    # reassemble sentences
    # sentences = [' '.join(s) for s in trainset]
    sentences = trainset[:n_sentences]

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        shuffle(sentences)
        fnet.enter_batch(sentences) # send all sentences for training



if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Using device: {device}')


    load_dotenv()
    asset_dir = os.getenv('ASSET_DIR', default='assets')
    corpus_file = "100k.txt"
    corpus_path = os.path.join(asset_dir, corpus_file)

    model_dir = os.getenv('MODEL_DIR', default='models')
    model_file = 'gpt_neox_10k.pt'
    model_path = os.path.join(model_dir, model_file )


    if 'gpt_neox' in model_file:
        fnet = GPTNeoXFNet(batch_size=75)
    else:
        raise Exception('Unknown model type')

    if os.path.exists(model_path):
        logger.info(f'loading model from: {model_path}')
        fnet.load(model_path)

    train_on_file(fnet, corpus_path, epochs=3)
    fnet.save(model_path)