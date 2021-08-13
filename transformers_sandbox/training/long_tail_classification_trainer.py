import torch.nn.functional as F
import torch
from transformers import Trainer


class LongTailClassificationTrainer(Trainer):
    """Long-Tail Learning via Logit Adjustment.

    https://openreview.net/forum?id=37nvvqkCo5

    """
    def set_prior(self, prior: torch.Tensor, tau: float=1.0):
        self.prior = prior
        self.tau = tau

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        # [2] => [1, 2]
        prior = torch.log(self.prior.to(model.device).unsqueeze(dim=0))
        # [batch_size, num_labels]
        logits = outputs.logits + (self.tau * prior)
        loss = F.cross_entropy(logits, labels)
        return (loss, outputs) if return_outputs else loss
