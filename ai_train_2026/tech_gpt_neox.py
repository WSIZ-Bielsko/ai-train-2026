from abc import ABC, abstractmethod
from typing import Iterable, Collection, List, Optional

import torch
from loguru import logger
from pydantic import BaseModel
from torch.utils.data import DataLoader

import torch.nn.functional as F
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
)
import random

from ai_train_2026.common import ts


# ==========================================================
# GPT-NeoX-backed implementation
# ==========================================================

class PredictionConfig(BaseModel):
    max_new_tokens: int = 5

class LmAPI(ABC):
    def enter_batch(self, corpora: list[str]):
        """ Method for training; enter a list of sentences/paragraphs. """

    def predict(self, prompt: str, config: PredictionConfig) -> str:
        """ Method for prediction; generate text based on a prompt. """


class GPTNeoXFNet(LmAPI):

    def __init__(self, device: str = None, batch_size = 60):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        # Standard tokenizer for GPT-NeoX and Pythia models
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Train a GPT-NeoX model from scratch (~300M parameters)
        config = GPTNeoXConfig(
            vocab_size=self.tokenizer.vocab_size,
            hidden_size=1024,
            intermediate_size=4096,
            num_hidden_layers=16,
            num_attention_heads=16,
            max_position_embeddings=512,
        )
        self.model = GPTNeoXForCausalLM(config)
        self.model.to(self.device)

        self.optimizer = AdamW(self.model.parameters(), lr=5e-5)

        self._known_labels = set()

    # ------------------------------------------------------
    # Core
    # ------------------------------------------------------

    def __train_on_data(self, data: str | list[str], run_alert=False):
        self.model.train()
        text = data

        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss

        if torch.isnan(loss):
            return

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()

        if run_alert:
            print(f"Training loss: {loss.item()}")


    def enter_batch(self, corpora: list[str]):
        bs = self.batch_size
        corpora = list(corpora)
        n_batches = (len(corpora) + bs - 1) // bs

        nst = ts()
        estimation_batch_sample = 100

        for i in range(n_batches):
            alert = False
            if i == estimation_batch_sample:
                duration = ts() - nst
                logger.warning(f'Full epoch training duration (estimation):'
                               f' {duration * n_batches/estimation_batch_sample:.1f} seconds')
            if i % 10 == 0:
                logger.info(f"Training on batch {i + 1} of {n_batches}")
                alert = True
            self.__train_on_data(corpora[i * bs: (i + 1) * bs], run_alert=alert)

    def predict(self, prompt: str, config=PredictionConfig()) -> str:
        self.model.eval()
        prompt = prompt.strip()
        # print(f'prompt: {prompt}')

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=False
            )
            #
            # output_ids = self.model.generate(
            #     **inputs,
            #     max_new_tokens=config.max_new_tokens,
            #     pad_token_id=self.tokenizer.pad_token_id,
            #     num_beams=5,  # Enables beam search
            #     do_sample=True,  # Required for temperature
            #     temperature=0.2  # Adjust as needed
            # )

        input_length = inputs["input_ids"].shape[1]
        new_ids = output_ids[0][input_length:]

        predicted_tokens = []
        for token_id in new_ids:
            token = self.tokenizer.decode(token_id).strip()
            predicted_tokens.append(token)
            if token in ['.', '?', '!']:
                break

        sentence = ' '.join(predicted_tokens)
        return sentence

    def save(self, filename):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, filename)


    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.model.to(self.device)
