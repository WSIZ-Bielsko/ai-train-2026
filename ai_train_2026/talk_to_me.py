import os
import socket

import torch
from dotenv import load_dotenv
from loguru import logger

from ai_train_2026.tech_gpt_neox import GPTNeoXFNet, PredictionConfig

if __name__ == '__main__':
    logger.info(f'Inference on host: {socket.gethostname()} ')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Using device: {device}')

    load_dotenv()

    model_dir = os.getenv('MODEL_DIR', default='models')
    model_file = 'gpt_neox_10k.pt'

    model_file = os.path.join(model_dir,model_file)

    logger.info(f'Loading model from: {model_file}')

    if 'gpt' in model_file:
        fnet = GPTNeoXFNet()
    else:
        raise Exception('Unknown model type')
    fnet.load(model_file)

    s = ''
    while s != 'x':
        s = input('> ')
        generated = fnet.predict(
            prompt=s,
            config=PredictionConfig(max_new_tokens=30)
        )
        print(s + " " + " ".join(generated))
        print('---')

