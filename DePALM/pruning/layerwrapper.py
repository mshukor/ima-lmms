import torch
import torch.nn as nn


# Define WrappedGPT class
class WrappedGPT:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples


# Define WrappedGPT class
class WrappedGPTLogits:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(
        self,
        layer,
        layer_id=0,
        layer_name="none",
        lm_head=None,
        layer_norm=None,
        indices=None,
        vocab_indices=None,
    ):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id
        self.layer_name = layer_name

        self.lm_head = lm_head
        self.layer_norm = layer_norm

        self.indices = indices  # for all elements in the batch
        self.vocab_indices = vocab_indices
        self.batch_idx = 0  # to know which element in the batch

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        # if isinstance(self.layer, nn.Linear):
        #     if len(inp.shape) == 3:
        #         inp = inp.reshape((-1, inp.shape[-1]))
        #     inp = inp.t()

        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp

        # inp = inp.type(torch.float32)

        logits = self.lm_head(self.layer_norm(inp))  # torch.norm(inp, p=2, dim=1) ** 2
        filtered_logits = []
        for idx, logit in zip(self.indices[self.batch_idx], logits):  # over batch
            useful_logit = logit[idx]  # (l, V)
            print(useful_logit.shape)
            if self.vocab_indices is not None:
                useful_logit = useful_logit[
                    :, torch.tensor(self.vocab_indices[self.batch_idx])
                ]
            else:
                useful_logit = torch.max(useful_logit, dim=-1).values
                print(useful_logit.shape, "max")
            filtered_logits.append(useful_logit)
        print(
            len(filtered_logits),
            len(self.indices),
            self.indices[self.batch_idx],
            logits.shape,
        )
        filtered_logits = torch.stack(filtered_logits)

        print(filtered_logits.shape, filtered_logits[0].shape)
        score = torch.norm(filtered_logits, p=2, dim=1) ** 2

        print(score, useful_logit.shape, idx, torch.norm(inp, p=2, dim=1) ** 2)
        self.scaler_row += score / self.nsamples

        self.batch_idx += 1
