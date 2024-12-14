from transformers import AutoModelForSequenceClassification
import torch.nn as nn

class DistributedModel(nn.Module):
    def __init__(self, model_name, num_labels=2):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs
