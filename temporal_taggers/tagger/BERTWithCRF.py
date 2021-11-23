"""
Implementation of BERT with a CRF layer
Code adapted form https://github.com/Louis-udm/NER-BERT-CRF/blob/master/NER_BERT_CRF.py
"""

from transformers import BertModel
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.bert.modeling_bert import BertPreTrainedModel
import torch
from torch import nn
from torch.nn import LayerNorm

# Hack to guarantee backward-compatibility.
BertLayerNorm = LayerNorm


def log_sum_exp_batch(log_Tensor, axis=-1):
    """
    Expected shape: (batch_size, n, m)
    """
    return torch.max(log_Tensor, axis)[0] + \
           torch.log(torch.exp(log_Tensor - torch.max(log_Tensor, axis)[0].view(log_Tensor.shape[0], -1, 1)).sum(axis))


class BertWithCRF(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.hidden_size = 768
        self.start_label_id = config.start_label_id
        self.stop_label_id = config.stop_label_id
        self.num_labels = config.num_classes
        self.batch_size = config.batch_size

        # Pre-trainded BertModel
        self.bert = BertModel(config, add_pooling_layer=False)

        self.dropout = torch.nn.Dropout(0.2)
        # Maps the output of BERT into label space.
        self.hidden2label = nn.Linear(self.hidden_size, self.num_labels)

        # Matrix of transition parameters. Entry [i,j] is the score of transitioning *to* i *from* j.
        self.transitions = nn.Parameter(torch.randn(self.num_labels, self.num_labels))

        # These two statements enforce the constraint that we never transfer *to* the start tag(or label),
        # and we never transfer *from* the stop label (the model would probably learn this anyway,
        # so this enforcement is likely unimportant)

        self.transitions.data[self.start_label_id, :] = -10000
        self.transitions.data[:, self.stop_label_id] = -10000

        nn.init.xavier_uniform_(self.hidden2label.weight)
        nn.init.constant_(self.hidden2label.bias, 0.0)

    def _forward_alg(self, feats):
        """
        This is also called "alpha-recursion" or "forward recursion", to calculate the log probability of all barX
        """

        # T = self.max_seq_length
        T = feats.shape[1]
        batch_size = feats.shape[0]

        # alpha_recursion,forward, alpha(zt)=p(zt,bar_x_1:t)
        log_alpha = torch.Tensor(batch_size, 1, self.num_labels).fill_(-10000.).to(self.device)
        # normal_alpha_0 : alpha[0]=Ot[0]*self.PIs
        # self.start_label has all of the score. it is log,0 is p=1
        log_alpha[:, 0, self.start_label_id] = 0

        # feats: sentences -> word embedding -> lstm -> MLP -> feats
        # feats is the probability of emission, feat.shape=(1,tag_size)
        for t in range(1, T):
            log_alpha = (log_sum_exp_batch(self.transitions + log_alpha, axis=-1) + feats[:, t]).unsqueeze(1)

        # log_prob of all barX
        log_prob_all_barX = log_sum_exp_batch(log_alpha)
        return log_prob_all_barX

    def _get_bert_features(self, input_ids,
                           attention_mask,
                           token_type_ids,
                           position_ids,
                           head_mask,
                           inputs_embeds,
                           output_attentions,
                           output_hidden_states,
                           return_dict):
        """
        sentences -> word embedding -> lstm -> MLP -> feats
        """
        bert_seq_out = self.bert(input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids,
                                 position_ids=position_ids,
                                 head_mask=head_mask,
                                 inputs_embeds=inputs_embeds,
                                 output_attentions=output_attentions,
                                 output_hidden_states=output_hidden_states,
                                 return_dict=return_dict)  # output_all_encoded_layers=False removed

        bert_seq_out_last = bert_seq_out[0]
        bert_seq_out_last = self.dropout(bert_seq_out_last)
        bert_feats = self.hidden2label(bert_seq_out_last)
        return bert_feats, bert_seq_out

    def _score_sentence(self, feats, label_ids):
        """
        Gives the score of a provided label sequence
        p(X=w1:t,Zt=tag1:t)=...p(Zt=tag_t|Zt-1=tag_t-1)p(xt|Zt=tag_t)...
        """

        # T = self.max_seq_length
        T = feats.shape[1]
        batch_size = feats.shape[0]

        batch_transitions = self.transitions.expand(batch_size, self.num_labels, self.num_labels)
        batch_transitions = batch_transitions.flatten(1)

        score = torch.zeros((feats.shape[0], 1)).to(self.device)
        # the 0th node is start_label->start_word, the probability of them=1. so t begins with 1.
        for t in range(1, T):
            score = score + \
                    batch_transitions.gather(-1,
                                             (label_ids[:, t] * self.num_labels + label_ids[:, t - 1]).view(-1, 1)) + \
                    feats[:, t].gather(-1, label_ids[:, t].view(-1, 1)).view(-1, 1)
        return score

    def _viterbi_decode(self, feats):
        """
        Max-Product Algorithm or Viterbi algorithm; argmax(p(z_0:t|x_0:t))
        """
        T = feats.shape[1]
        batch_size = feats.shape[0]
        log_delta = torch.Tensor(batch_size, 1, self.num_labels).fill_(-10000.).to(self.device)
        log_delta[:, 0, self.start_label_id] = 0

        # psi is for the value of the last latent that make P(this_latent) maximum.
        psi = torch.zeros((batch_size, T, self.num_labels), dtype=torch.long).to(self.device)  # psi[0]=0000 useless
        for t in range(1, T):
            log_delta, psi[:, t] = torch.max(self.transitions + log_delta, -1)

            log_delta = (log_delta + feats[:, t]).unsqueeze(1)

        # trace back
        path = torch.zeros((batch_size, T), dtype=torch.long).to(self.device)

        # max p(z1:t,all_x|theta)
        max_logLL_allz_allx, path[:, -1] = torch.max(log_delta.squeeze(), -1)

        for t in range(T - 2, -1, -1):
            # choose the state of z_t according the state chosen of z_t+1.
            path[:, t] = psi[:, t + 1].gather(-1, path[:, t + 1].view(-1, 1)).squeeze()

        return max_logLL_allz_allx, path

    def neg_log_likelihood(self, input_ids,
                           attention_mask,
                           token_type_ids,
                           position_ids,
                           head_mask,
                           inputs_embeds,
                           output_attentions,
                           output_hidden_states,
                           return_dict,
                           label_ids):

        bert_feats, _ = self._get_bert_features(input_ids,
                                                attention_mask,
                                                token_type_ids,
                                                position_ids,
                                                head_mask,
                                                inputs_embeds,
                                                output_attentions,
                                                output_hidden_states,
                                                return_dict)

        forward_score = self._forward_alg(bert_feats)
        # p(X=w1:t,Zt=tag1:t)=...p(Zt=tag_t|Zt-1=tag_t-1)p(xt|Zt=tag_t)...
        gold_score = self._score_sentence(bert_feats, label_ids)
        # - log[ p(X=w1:t,Zt=tag1:t)/p(X=w1:t) ] = - log[ p(Zt=tag1:t|X=w1:t) ]
        return torch.mean(forward_score - gold_score)

    # this forward is just for predict, not for train
    # dont confuse this with _forward_alg above.
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            inference_mode=False,
    ):
        # Get the emission scores from BERT
        bert_feats, bert_out = self._get_bert_features(input_ids,
                                                       attention_mask,
                                                       token_type_ids,
                                                       position_ids,
                                                       head_mask,
                                                       inputs_embeds,
                                                       output_attentions,
                                                       output_hidden_states,
                                                       return_dict)

        # Find the best path, given the features
        score, label_seq_ids = self._viterbi_decode(bert_feats)

        if not inference_mode:
            neg_log_likelihood = self.neg_log_likelihood(input_ids,
                                                         attention_mask,
                                                         token_type_ids,
                                                         position_ids,
                                                         head_mask,
                                                         inputs_embeds,
                                                         output_attentions,
                                                         output_hidden_states,
                                                         return_dict,
                                                         labels)

            return TokenClassifierOutput(
                loss=neg_log_likelihood,
                logits=label_seq_ids,
                hidden_states=bert_out.hidden_states,
                attentions=bert_out.attentions,
            )

        else:
            neg_log_likelihood = None
            return TokenClassifierOutput(
                loss=neg_log_likelihood,
                logits=label_seq_ids,
                hidden_states=bert_out.hidden_states,
                attentions=bert_out.attentions,
            )
