import torch
from torch import nn, Tensor, optim
from transformers import AutoTokenizer, AutoModel

from kge.model.kge_model import KgeModel

class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(config.get("encoder.config"))
        self.tokenizer = AutoTokenizer.from_pretrained(config.get("encoder.config"))
        self.proj = nn.Linear(
            self.encoder.config.hidden_size, 
            config.get("lookup_embedder.dim"),     # 128
        )
        self.device = 'cpu'

    def tokenize(self, x):
        return self.tokenizer(x, return_tensors="pt", padding=True)

    def forward(self, x):
        out = self.encoder(**x)
        out = out["last_hidden_state"][:,0]     # Use CLS token only
        out = self.proj(out)
        return out

    def __call__(self, inputs: list):
        idxs = torch.cat(inputs, axis=0)
        x = self.prepare_inputs(idxs)
        out = self.forward(x)
        outs = torch.split(out, inputs[0].size(0))
        return outs[0] if len(outs) == 1 else outs
        # return torch.split(out, inputs[0].size(0))

    def prepare_entity_ids(self, dataset):
        entity_names = [ent[1] for ent in dataset._meta['entity_ids']]
        self.entity_toks = self.tokenize(entity_names)

    def prepare_inputs(self, idxs):
        x = {}
        for k,v in self.entity_toks.items():
            x[k] = v[idxs.long()].to(self.device)
        return x

    def to(self, device):
        self.device = device
        super().to(device)
        return self


class EncodedKge(KgeModel):
    def __init__(self, config, kge: KgeModel):
        self.__dict__ = kge.__dict__
        self.config = config
        self.encoder = BertEncoder(config)
        
    def score_spo(self, s: Tensor, p: Tensor, o: Tensor, direction=None) -> Tensor:
        ''' Overloads KgeModel.score_spo '''
        if True:
            s, o = self.encoder([s, o])
        else:
            s = self.get_s_embedder().embed(s)
            o = self.get_o_embedder().embed(o)
        p = self.get_p_embedder().embed(p)
        return self._scorer.score_emb(s, p, o, combine="spo").view(-1)

    def score_sp_po(
        self, s: Tensor, p: Tensor, o: Tensor, entity_subset: Tensor = None
    ) -> Tensor:
        r"""Combine `score_sp` and `score_po`.

        `s`, `p` and `o` are vectors of common size :math:`n`, holding the indexes of
        the subjects, relations, and objects to score.

        Each sp-pair and each po-pair is scored against the entities in `entity_subset`
        (also holds indexes). If set to `entity_subset` is `None`, scores against all
        entities.

        The result is the horizontal concatenation of the outputs of
        :code:`score_sp(s,p,entity_subset)` and :code:`score_po(p,o,entity_subset)`.
        I.e., returns an :math:`n\times 2E` tensor, where :math:`E` is the size of
        `entity_subset`. For :math:`j<E`, the :math:`(i,j)`-entry holds the score for
        triple :math:`(s_i, p_i, e_j)`. For :math:`j\ge E`, the :math:`(i,j)`-entry
        holds the score for triple :math:`(e_{j-E}, p_i, o_i)`.

        """
        if True:
            s, o = self.encoder([s, o])
        else:
            s = self.get_s_embedder().embed(s)
            o = self.get_o_embedder().embed(o)
        p = self.get_p_embedder().embed(p)

        # TODO: check parent class score_sp_po if this fails

        if entity_subset is not None:
            all_entities = self.encoder([entity_subset])
        else:
            raise NotImplementedError
        sp_scores = self._scorer.score_emb(s, p, all_entities, combine="sp_")
        po_scores = self._scorer.score_emb(all_entities, p, o, combine="_po")
        return torch.cat((sp_scores, po_scores), dim=1)

    def score_so(self, *args):
        raise NotImplementedError

    def score_sp(self, *args):
        raise NotImplementedError

    def score_o(self, *args):
        raise NotImplementedError

    def to(self, device):
        super().to(device)
        self.device = device
        self.encoder = self.encoder.to(device)
        return self

    def train(self, turn_on=True):
        self.encoder.train(turn_on)
        super().train(turn_on)

    def eval(self):
        self.train(False)

class CompoundOptimizer(optim.Optimizer):
    def __init__(self, optimizers: list):
        super().__init__()
        self.optimizers = optimizers

    def step(self):
        for opt in self.optimizers:
            opt.step()
    
    def zero_grad(self):
        for opt in self.optimizers:
            opt.zero_grad()


if __name__ == '__main__':
    bert = BertEncoder()
    txt = ['hello world', 'what a nice day']
    # inputs = bert.tokenize(txt)
    # print(bert(inputs).shape)
    print(bert(txt).shape)
    