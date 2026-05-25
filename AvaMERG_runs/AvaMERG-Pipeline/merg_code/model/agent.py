import os
import datetime
from collections import OrderedDict

import torch
from torch.utils.tensorboard import SummaryWriter

from header import *


class DummySummaryWriter:
    """Fallback writer that safely ignores TensorBoard calls."""

    def add_scalar(self, *args, **kwargs):
        pass

    def add_text(self, *args, **kwargs):
        pass

    def add_image(self, *args, **kwargs):
        pass

    def add_histogram(self, *args, **kwargs):
        pass

    def flush(self):
        pass

    def close(self):
        pass


class LocalEngine:
    def __init__(self, model, optimizer, device):
        self.module = model
        self.optimizer = optimizer
        self.device = device

    def __call__(self, batch):
        return self.module(batch)

    def backward(self, loss):
        loss.backward()

    def step(self):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def generate(self, args):
        return self.module.generate(args)


class DeepSpeedAgent:

    def __init__(self, model, args):
        super(DeepSpeedAgent, self).__init__()
        self.args = args
        self.model = model

        self.print_model_parameters()
        self.writer = self._build_writer()

        if torch.cuda.is_available():
            local_rank = args.get("local_rank", 0)
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device("cpu")

        self.model = self.model.to(self.device)

        if self.args['mode'] == 'test':
            self.load_parameters(self.args['save_path'])

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=5e-6,
            betas=(0.9, 0.95),
            eps=1e-3,
            weight_decay=1e-3
        )

        self.ds_engine = LocalEngine(self.model, self.optimizer, self.device)

    def _build_writer(self):
        """
        Try the configured log_path first.
        If it is not writable, fall back to a user-owned path.
        If both fail, disable TensorBoard safely.
        """
        fallback_log_path = "/scratch/e1554543/avatar_system_full/AvaMERG_runs/AvaMERG-Pipeline/outputs/tb_logs"

        candidate_paths = []
        if self.args.get("log_path"):
            candidate_paths.append(self.args["log_path"])
        candidate_paths.append(fallback_log_path)

        seen = set()
        unique_paths = []
        for p in candidate_paths:
            if p and p not in seen:
                unique_paths.append(p)
                seen.add(p)

        for log_path in unique_paths:
            try:
                os.makedirs(log_path, exist_ok=True)
                writer = SummaryWriter(log_path)
                self.args["log_path"] = log_path
                print(f"[INFO] TensorBoard log path: {log_path}")
                return writer
            except Exception as e:
                print(f"[WARN] Failed to initialize TensorBoard at {log_path}: {e}")

        print("[WARN] TensorBoard disabled.")
        self.args["log_path"] = None
        return DummySummaryWriter()

    @torch.no_grad()
    def predict(self):
        self.ds_engine.module.eval()
        output = self.ds_engine.generate(self.args)
        return output

    def train_model(self, batch, current_step=0, pbar=None):
        self.ds_engine.module.train()
        loss_dict = self.ds_engine(batch)

        for k, v in loss_dict.items():
            if torch.is_tensor(v):
                self.writer.add_scalar(k, v.item(), current_step)
            else:
                self.writer.add_scalar(k, v, current_step)

        loss = loss_dict['loss']
        if 'gen_acc' in loss_dict.keys():
            mle_acc = loss_dict['gen_acc']
            if torch.is_tensor(mle_acc):
                mle_acc = mle_acc.item()
        else:
            mle_acc = 0

        self.ds_engine.backward(loss)
        self.ds_engine.step()

        if torch.is_tensor(loss):
            loss_value = loss.item()
        else:
            loss_value = float(loss)

        if pbar is not None:
            pbar.set_description(f'[!] loss: {round(loss_value, 4)}; token_acc: {round(mle_acc * 100, 2)}')
            pbar.update(1)

            if self.args.get('log_path') and current_step % self.args['logging_step'] == 0:
                elapsed = pbar.format_dict.get('elapsed', 0)
                rate = pbar.format_dict.get('rate', None)
                remaining = (pbar.total - pbar.n) / rate if rate and pbar.total else 0
                remaining = str(datetime.timedelta(seconds=remaining))
                logging.info(
                    f'[!] progress: {round(pbar.n / pbar.total, 5)}; remaining time: {remaining}; loss: {round(loss_value, 4)}; token_acc: {round(mle_acc * 100, 2)}'
                )

        mle_acc *= 100
        return mle_acc

    def save_model(self, path, epoch, current_step):
        path = os.path.join(path, f'{epoch}')

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        checkpoint = OrderedDict()
        for k, v in self.ds_engine.module.named_parameters():
            if v.requires_grad:
                checkpoint[k] = v.detach().cpu()
            if 'llama_proj' in k:
                checkpoint[k] = v.detach().cpu()

        torch.save(checkpoint, f'{path}/pytorch_model.pt')
        self.model.llama_tokenizer.save_pretrained(path)
        self.model.llama_model.config.save_pretrained(path)
        print(f'[!] save model into {path}')

    def print_model_parameters(self, use_4bit=False):
        trainable_params = 0
        all_param = 0
        lora = 0
        ccl = 0
        sdm = 0
        linear = 0
        llama = 0
        imagebind = 0
        for name, param in self.model.named_parameters():
            num_params = param.numel()

            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            if 'lora' in name:
                lora += num_params
            elif 'llama_proj' in name:
                linear += num_params
            elif 'llama_model' in name:
                llama += num_params
            elif 'visual_encoder' in name:
                imagebind += num_params

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        if use_4bit:
            trainable_params /= 2

        print(f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}")
        print(f'lora params: {lora:,d} || ccl params: {ccl:,d} || sdm params: {sdm:,d}')
        print(f'linear params: {linear:,d} || imagebind params: {imagebind:,d} || llama params: {llama:,d}')

    def load_parameters(self, path):
        if os.path.exists(os.path.join(path, 'pytorch_model.pt')):
            print('#########################################################')
            print('loading parameters from {}'.format(path))
            print('#########################################################')
            delta_ckpt = torch.load(f'{path}/pytorch_model.pt', map_location=self.device)
            checkpoint = OrderedDict()
            checkpoint = delta_ckpt
            self.model.load_state_dict(checkpoint, strict=False)