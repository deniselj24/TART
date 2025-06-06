import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Config,
    GPT2Model,
    GPTNeoForCausalLM,
)


def build_model(conf):
    if conf.family == "gpt2":
        model = TransformerModel(
            n_dims=conf.n_dims,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
            n_y=conf.n_y,
            model_name=conf.model_name,
        )
    elif conf.family == "gpt-neo":
        model = TransformerLanguageModel(
            n_dims=conf.n_dims,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
            n_y=conf.n_y,
            model_name=conf.model_name,
            lr_solver_head=conf.lr_solver_head,
        )
    elif conf.family == "mlp":
        model = MLP(num_classes=conf.num_classes)
    elif conf.family == "logistic_regression":
        model = LogisticRegression()
    else:
        raise NotImplementedError

    return model



class MLP(nn.Module):
    def __init__(
        self,
        num_classes=2,  # number of classes (positive or negative)
        max_len=200,  # number of tokens # max for sms dataset is ~171 # max for RT Is 59 
        vocab_size=18484, #18484, #agnews: 72049, #rt: 18484, #sms: 8956,  # number of in-features / vocab size 
        hidden_features=[256, 256],  # number of hidden features
        embedding_dim=512,  # embedding dimension
        **kwargs,
    ):
        super(MLP, self).__init__()
        self.num_classes = num_classes
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.hidden_features = hidden_features
        self.embedding_dim = embedding_dim
        #self.use_token_embeddings = use_token_embeddings
        self.n_y = 1

        # Build network layers
        layers = []
        layers.extend(
            [
                nn.Linear(self.max_len, self.hidden_features[0]),
                nn.ReLU(),
                # nn.GELU(),
            ]
        )

        # Add remaining layers
        for i in range(len(hidden_features) - 1):
            layers.extend(
                [
                    nn.Linear(hidden_features[i], hidden_features[i + 1]),
                    nn.ReLU(), #nn.GELU(),
                ]
            )

        self.net = nn.Sequential(*layers)
        self.fc_out = nn.Linear(hidden_features[-1], self.num_classes)

    @staticmethod
    def _combine(xs_b: torch.Tensor, ys_b: torch.Tensor):
        """Interleaves the x's and the y's into a single sequence."""
        bsize, points, dim = xs_b.shape
        ys_b_wide = torch.cat(
            (
                ys_b.view(bsize, points, 1),
                torch.zeros(bsize, points, dim - 1, device=ys_b.device),
            ),
            axis=2,
        )
        zs = torch.stack((xs_b, ys_b_wide), dim=2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs

    def _combine_gen(self, xs_b: torch.Tensor, ys_b: torch.Tensor):
        """For sequences with more than one y's, Interleaves the x's
        and the y's into a single sequence."""
        bsize, points, dim = xs_b.shape
        ys_list = []
        for i in range(self.n_y):
            ys_b_i = ys_b[i, ::]
            ys_b_i_wide = torch.cat(
                (
                    ys_b_i.view(bsize, points, 1),
                    torch.zeros(bsize, points, dim - 1, device=ys_b.device),
                ),
                axis=2,
            )
            ys_list.append(ys_b_i_wide)
        zs = torch.stack((xs_b, *ys_list), dim=2)
        zs = zs.view(bsize, (self.n_y + 1) * points, dim)

        return zs

    def forward(self, xs: torch.Tensor, ys: torch.Tensor, inds=None):
        # Predicting a *sequence* of y's
        if len(ys.shape) > 2:
            print("ys shape", ys.shape)
            inds = torch.arange(ys.shape[-1])
            zs = self._combine_gen(xs, ys)

            output = self.net(zs)
            prediction = F.sigmoid(self.fc_out(output))

            preds_y = []
            for i in range(self.n_y):
                preds_y.append(prediction[:, i :: self.y_step_size, 0][:, inds])
            return preds_y
        # Predicting a single y
        else:
            # if predicting a single y
            if inds is None:
                inds = torch.arange(ys.shape[1])
            else:
                inds = torch.tensor(inds)
                if max(inds) >= ys.shape[1] or min(inds) < 0:
                    raise ValueError(
                        "inds contain indices where xs and ys are not defined"
                    )
            #print("xs", xs.shape)
            #print("ys", ys.shape)
            zs = self._combine(xs, ys)
            #print("zs", zs.shape)
            zs = zs.to('cuda')
            output = self.net(zs)
            #prediction = F.sigmoid(self.fc_out(output))
            prediction = self.fc_out(output)
            softmax = nn.Softmax(dim=-1)
            prediction = softmax(prediction)
            #print("model prediction shape", prediction.shape)
            #print("model prediction shape after", prediction[:, ::2, :self.num_classes].shape)
            return prediction[:, ::2, :self.num_classes][
                :
            ]  # return hiddens pertaining to x's indexes


class LogisticRegression(nn.Module):
    def __init__(
        self,
        num_classes=2,  # number of classes (positive or negative)
        max_len=200,  # number of tokens # max for sms dataset is ~171 
        vocab_size=8956,  # number of in-features / vocab size 
        hidden_features=[256, 256],  # number of hidden features
        embedding_dim=512,  # embedding dimension
        **kwargs,
    ):
        super(LogisticRegression, self).__init__()
        self.num_classes = num_classes
        self.max_len = max_len
        self.vocab_size = vocab_size
        #self.hidden_features = hidden_features
        #self.embedding_dim = embedding_dim
        #self.use_token_embeddings = use_token_embeddings
        self.n_y = 1

        # Build network layers
        layers = []
        layers.extend(
            [
                #nn.Linear(self.max_len, self.num_classes),
            ]
        )
        self.net = nn.Sequential(*layers)
        self.fc_out = nn.Linear(self.max_len, self.num_classes)

    @staticmethod
    def _combine(xs_b: torch.Tensor, ys_b: torch.Tensor):
        """Interleaves the x's and the y's into a single sequence."""
        bsize, points, dim = xs_b.shape
        ys_b_wide = torch.cat(
            (
                ys_b.view(bsize, points, 1),
                torch.zeros(bsize, points, dim - 1, device=ys_b.device),
            ),
            axis=2,
        )
        zs = torch.stack((xs_b, ys_b_wide), dim=2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs

    def _combine_gen(self, xs_b: torch.Tensor, ys_b: torch.Tensor):
        """For sequences with more than one y's, Interleaves the x's
        and the y's into a single sequence."""
        bsize, points, dim = xs_b.shape
        ys_list = []
        for i in range(self.n_y):
            ys_b_i = ys_b[i, ::]
            ys_b_i_wide = torch.cat(
                (
                    ys_b_i.view(bsize, points, 1),
                    torch.zeros(bsize, points, dim - 1, device=ys_b.device),
                ),
                axis=2,
            )
            ys_list.append(ys_b_i_wide)
        zs = torch.stack((xs_b, *ys_list), dim=2)
        zs = zs.view(bsize, (self.n_y + 1) * points, dim)

        return zs

    def forward(self, xs: torch.Tensor, ys: torch.Tensor, inds=None):
        # Predicting a *sequence* of y's
        if len(ys.shape) > 2:
            inds = torch.arange(ys.shape[-1])
            zs = self._combine_gen(xs, ys)
            #embeds = self.embeddings(zs)

            #output = self.net(
            #    zs,
            #)
            #prediction = F.sigmoid(self.fc_out(output))
            prediction = F.sigmoid(self.fc_out(zs))

            preds_y = []
            for i in range(self.n_y):
                preds_y.append(prediction[:, i :: self.y_step_size, 0][:, inds])
            return preds_y
        # Predicting a single y
        else:
            # if predicting a single y
            if inds is None:
                inds = torch.arange(ys.shape[1])
            else:
                inds = torch.tensor(inds)
                if max(inds) >= ys.shape[1] or min(inds) < 0:
                    raise ValueError(
                        "inds contain indices where xs and ys are not defined"
                    )
            zs = self._combine(xs, ys)
            #embeds = self.embeddings(zs)
            #output = self.net(zs)
            #prediction = F.sigmoid(self.fc_out(output))
            prediction = F.sigmoid(self.fc_out(zs))

            return prediction[:, ::2, 0][
                :, inds
            ]  # return hiddens pertaining to x's indexe


class TransformerLanguageModel(nn.Module):
    def __init__(
        self,
        n_dims: int,
        n_positions: int,
        n_embd: int = 128,
        n_layer: int = 12,
        n_head: int = 4,
        n_y: int = 1,
        model_name="EleutherAI/gpt-neo-125M",
    ):
        super(TransformerLanguageModel, self).__init__()
        model_name_short = model_name.split("/")[-1]

        self.name = f"{model_name_short}_embd={n_embd}_layer={n_layer}_head={n_head}"
        self.n_positions = n_positions
        self.n_dims = n_dims
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.positive_token_id_space = self._tokenizer(" positive").input_ids[0]
        self.negative_token_id_space = self._tokenizer(" negative").input_ids[0]

        if "pythia" in model_name:
            print(f"loading pythia model: {model_name}")
            self._backbone = AutoModelForCausalLM.from_pretrained(
                model_name, cache_dir="/u/scr/nlp/data/neo/hub"
            )
            for param in self._backbone.gpt_neox.embed_in.parameters():
                param.requires_grad = False
            for param in self._backbone.embed_out.parameters():
                param.requires_grad = False
        elif "opt" in model_name:
            print(f"loading opt model: {model_name}")
            self._backbone = AutoModelForCausalLM.from_pretrained(
                model_name, cache_dir="/u/scr/nlp/data/neo/hub"
            )
            for param in self._backbone.model.decoder.embed_tokens.parameters():
                param.requires_grad = False
            for param in self._backbone.lm_head.parameters():
                param.requires_grad = False
        else:
            print(f"loading GPT-Neo model: {model_name}")
            self._backbone = GPTNeoForCausalLM.from_pretrained(
                model_name, cache_dir="/u/scr/nlp/data/neo/hub"
            )
            for param in self._backbone.transformer.wte.parameters():
                param.requires_grad = False
            for param in self._backbone.lm_head.parameters():
                param.requires_grad = False

        self.y_step_size = n_y + 1
        self.n_y = n_y

    def forward(
        self,
        xs,
        ys,
    ):
        output = self._backbone(input_ids=xs).logits
        return output


class TransformerModel(nn.Module):
    def __init__(
        self,
        n_dims: int,
        n_positions: int,
        n_embd: int = 128,
        n_layer: int = 12,
        n_head: int = 4,
        n_y: int = 1,
        model_name: str = None,
    ):
        super(TransformerModel, self).__init__()
        configuration = GPT2Config(
            n_positions=(n_y + 1) * n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"

        self.n_positions = n_positions
        self.n_dims = n_dims
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, 1)
        self.y_step_size = n_y + 1
        self.n_y = n_y
        self.sigmoid = torch.nn.Sigmoid()

    @staticmethod
    def _combine(xs_b: torch.Tensor, ys_b: torch.Tensor):
        """Interleaves the x's and the y's into a single sequence."""
        bsize, points, dim = xs_b.shape
        ys_b_wide = torch.cat(
            (
                ys_b.view(bsize, points, 1),
                torch.zeros(bsize, points, dim - 1, device=ys_b.device),
            ),
            axis=2,
        )
        zs = torch.stack((xs_b, ys_b_wide), dim=2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs

    def _combine_gen(self, xs_b: torch.Tensor, ys_b: torch.Tensor):
        """For sequences with more than one y's, Interleaves the x's
        and the y's into a single sequence."""
        bsize, points, dim = xs_b.shape
        ys_list = []
        for i in range(self.n_y):
            ys_b_i = ys_b[i, ::]
            ys_b_i_wide = torch.cat(
                (
                    ys_b_i.view(bsize, points, 1),
                    torch.zeros(bsize, points, dim - 1, device=ys_b.device),
                ),
                axis=2,
            )
            ys_list.append(ys_b_i_wide)
        zs = torch.stack((xs_b, *ys_list), dim=2)
        zs = zs.view(bsize, (self.n_y + 1) * points, dim)

        return zs

    def _step(self, zs: torch.Tensor):
        inds = torch.arange(int(zs.shape[1] / 2))
        embeds = self._read_in(zs)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        y_outs = self._read_out(output)

        predictions = y_outs[:, ::2, 0][:, inds]
        return predictions

    def predict(self, zs: torch.Tensor):
        inds = torch.arange(int(zs.shape[1] / 2))
        embeds = self._read_in(zs)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        y_outs = self._read_out(output)

        predictions = y_outs[:, ::2, 0][:, inds]
        pred = self.sigmoid(predictions)[0][-1].item()

        if pred >= 0.5:
            return 1
        else:
            return 0

    def forward(self, xs: torch.Tensor, ys: torch.Tensor, inds=None):
        # Predicting a *sequence* of y's
        if len(ys.shape) > 2:
            inds = torch.arange(ys.shape[-1])
            zs = self._combine_gen(xs, ys)
            embeds = self._read_in(zs)

            output = self._backbone(
                inputs_embeds=embeds,
            ).last_hidden_state
            prediction = self._read_out(output)

            preds_y = []
            for i in range(self.n_y):
                preds_y.append(prediction[:, i :: self.y_step_size, 0][:, inds])
            return preds_y
        # Predicting a single y
        else:
            # if predicting a single y
            if inds is None:
                inds = torch.arange(ys.shape[1])
            else:
                inds = torch.tensor(inds)
                if max(inds) >= ys.shape[1] or min(inds) < 0:
                    raise ValueError(
                        "inds contain indices where xs and ys are not defined"
                    )
            zs = self._combine(xs, ys)
            embeds = self._read_in(zs)
            output = self._backbone(inputs_embeds=embeds).last_hidden_state
            prediction = self._read_out(output)
            return prediction[:, ::2, 0][
                :, inds
            ]  # return hiddens pertaining to x's indexes
