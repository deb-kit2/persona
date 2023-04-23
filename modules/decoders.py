import torch.nn as nn

# BART
from transformers.models.bart.modeling_bart import BartDecoder, BartPretrainedModel
from transformers import BartConfig

# T5

class BARTDecoder(BartPretrainedModel) :
  def __init__(self, config: BartConfig) :
    super().__init__(config)

    padding_idx, vocab_size = config.pad_token_id, config.vocab_size
    self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

    self.decoder = BartDecoder(config, self.shared)
  
  def forward(self, input_ids, attention_mask = None, 
              output_attentions = False, output_hidden_states = False, return_dict = False) :

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    decoder_outputs = self.decoder(
        input_ids = input_ids,
        attention_mask = attention_mask,
        output_attentions = output_attentions,
        output_hidden_states = output_hidden_states,
        return_dict = return_dict,
    )

    return decoder_outputs


class T5Decoder() :
    pass