import torch
from loguru import logger
import socket


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'host: {socket.gethostname()}')
    logger.info(f'device: {device}')
