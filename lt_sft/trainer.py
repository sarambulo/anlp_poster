import torch
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from typing import Dict, Optional, Tuple, Union, Any
from torch import nn
from transformers.utils import is_sagemaker_mp_enabled
# from transformers.trainer_pt_utils import smp_forward_backward

class LTSFTSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, device='cuda' if torch.cuda.is_available() else 'cpu', param_masks: Optional[Dict[str, torch.Tensor]] = None, **kwargs):
        """
        Initializes the ElementWiseFreezeSeq2SeqTrainer.

        Args:
            *args:  Positional arguments passed to Seq2SeqTrainer.
            param_masks (Optional[Dict[str, torch.Tensor]]):
                A dictionary mapping parameter names to their corresponding mask tensors.
                The mask tensors should have the same shape as the parameter tensors.
                A value of 0 in the mask indicates that the gradient for that parameter
                element should be frozen (set to zero) during training.  A value of 1
                indicates the gradient should be updated.
            **kwargs: Keyword arguments passed to Seq2SeqTrainer.
        """
        super().__init__(*args, **kwargs)
        self.device = device
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.param_masks = param_masks if param_masks is not None else {}
        # Ensure masks are on the correct device
        for name, mask in self.param_masks.items():
            self.param_masks[name] = mask.to(device)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            raise ValueError("We do not have smp_forward_backward")
            # loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            # return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            raise ValueError("We do not have apex")
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        # ====================== CUSTOM GRADIENT MASKING ======================
        if self.param_masks:
            with torch.no_grad():  # Crucial:  Don't track gradients during masking!
                for name, param in model.named_parameters():
                    if name in self.param_masks and param.grad is not None:
                        mask = self.param_masks[name]
                        # Ensure mask and gradient are on the same device.
                        if mask.device != param.grad.device:
                            mask = mask.to(param.grad.device)
                        param.grad *= mask  # Apply the element-wise mask

        # ==========================================================================

        return loss.detach()  # Return detached loss.