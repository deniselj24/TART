import math

import numpy as np
import torch
import os
import pandas as pd

# Data load function
def load_data(data_path, dataset="sms", input_key="sms", seed=42):
    # load training data
    train_set_path = os.path.join(data_path, f"{dataset}/train_samples_s{seed}.csv")
    training_set = pd.read_csv(train_set_path)
    X_train = training_set[input_key].tolist()
    y_train = training_set["label"].tolist()

    # load test data
    #if "ag_news" in dataset or "dbpedia" in dataset or "civil_comments" in dataset:
    #    test_set_path = os.path.join(data_path, f"{dataset}/test_samples_bal.csv")
    #else:
    test_set_path = os.path.join(data_path, f"{dataset}/test_samples_orig.csv")
    test_set = pd.read_csv(test_set_path)
    X_test = test_set[input_key].tolist()
    y_test = test_set["label"].tolist()

    # return
    return (X_train, y_train), (X_test, y_test)

class DataSampler:
    def __init__(self, n_dims: int):
        self.n_dims = n_dims

    def sample_xs(self):
        raise NotImplementedError


def get_data_sampler(data_name: str, n_dims: int, split: str = "train", **kwargs):
    names_to_classes = {
        "gaussian": GaussianSampler,
        "nl": NLSyntheticSampler,
        "sms": SMSDataSampler,
        "rt": RTDataSampler,
        "ag_news": AGDataSampler,
    }
    if data_name in ["gaussian", "nl"]:
        sampler_cls = names_to_classes[data_name]
        return sampler_cls(n_dims, **kwargs)
    elif data_name == "sms":
        return SMSDataSampler(n_dims=n_dims, split=split, **kwargs)
    elif data_name == "rt": 
        return RTDataSampler(n_dims=n_dims, split=split, **kwargs)
    elif data_name == "ag_news": 
        return AGDataSampler(n_dims=n_dims, split=split, **kwargs)
    else:
        print("Unknown sampler")
        raise NotImplementedError
         
def sample_transformation(eigenvalues, normalize=False):
    n_dims = len(eigenvalues)
    U, _, _ = torch.linalg.svd(torch.randn(n_dims, n_dims))
    t = U @ torch.diag(eigenvalues) @ torch.transpose(U, 0, 1)
    if normalize:
        norm_subspace = torch.sum(eigenvalues**2)
        t *= math.sqrt(n_dims / norm_subspace)
    return t


class NLSyntheticSampler(DataSampler):
    def __init__(self, n_dims: int, bias: float = None, scale: float = None):
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale

    def sample_xs(
        self, n_points: int, b_size: int, n_dims_truncated: int = None, seeds=None
    ):
        xs_b = np.random.choice([-1, 1], (b_size, n_points, self.n_dims))
        # set sample_sentence to a tensor of type double
        xs_b = torch.tensor(xs_b, dtype=torch.float32)
        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = -1
        return xs_b, None


class GaussianSampler(DataSampler):
    def __init__(self, n_dims: int, bias: float = None, scale: float = None):
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        if seeds is None:
            xs_b = torch.randn(b_size, n_points, self.n_dims)
        else:
            xs_b = torch.zeros(b_size, n_points, self.n_dims)
            generator = torch.Generator()
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                xs_b[i] = torch.randn(n_points, self.n_dims, generator=generator)
        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b, None

class SMSDataSampler(DataSampler):
    def __init__(self, n_dims: int, split: str = "train", bias: float = None, scale: float = None):
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale
        import json, re
        with open("src/reasoning_module/rt_vocabulary.json") as f:
            self.vocabulary = json.load(f) # dict mapping words to ids 
        self.vocabulary_size = len(self.vocabulary)
        print("vocab size", self.vocabulary_size)      
        self.max_len = 200  
        # Load the dataset
        (self.X_train, self.y_train), (self.X_test, self.y_test) = load_data(
            data_path="~/deniselj/Desktop/tart-denise/datasets/", dataset="rt", input_key="text", seed=42
        )
        
        def tokenize_and_pad(text: str) -> list:
            # converts to one-hot encoded vector of length 8955
            tokens = re.findall(r"[\w']+|[.,!?;:\"\'@#$%^&*()<>{}\[\]|\/\\~`+-=]", text.lower())
            tokens = tokens[:self.max_len] + ['<PAD>'] * (self.max_len - len(tokens))
            return [self.vocabulary.get(token) for token in tokens]

        # Convert to tensors
        self.num_train = len(self.X_train)
        self.num_test = len(self.X_test)
        self.X_train = torch.tensor([tokenize_and_pad(text) for text in self.X_train], dtype=torch.long)
        self.y_train = torch.tensor(self.y_train, dtype=torch.float32)

        self.X_test = torch.tensor([tokenize_and_pad(text) for text in self.X_test], dtype=torch.long)
        self.y_test = torch.tensor(self.y_test, dtype=torch.float32)
        
        self.split = split

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        # xs_b = torch.randn(b_size, n_points, self.n_dims)
        if self.split == "train":
            indices = torch.randint(0, self.num_train, (b_size, n_points))
        else: 
            indices = torch.randint(0, self.num_test, (b_size, n_points))
        if self.split == "train":
            xs_b = self.X_train[indices]
            ys_b = self.y_train[indices]
        else:
            xs_b = self.X_test[indices]
            ys_b = self.y_test[indices]
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b, ys_b


class RTDataSampler(DataSampler):
    def __init__(self, n_dims: int, split: str = "train", bias: float = None, scale: float = None):
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale
        import json, re
        with open("src/reasoning_module/rt_vocabulary.json") as f:
            self.vocabulary = json.load(f) # dict mapping words to ids 
        self.vocabulary_size = len(self.vocabulary)
        print("vocab size", self.vocabulary_size)      
        self.max_len = 200  
        # Load the dataset
        (self.X_train, self.y_train), (self.X_test, self.y_test) = load_data(
            data_path="~/deniselj/Desktop/tart-denise/datasets/", dataset="rt", input_key="text", seed=42
        )
        
        def tokenize_and_pad(text: str) -> list:
            # converts to one-hot encoded vector of length 8955
            tokens = re.findall(r"[\w']+|[.,!?;:\"\'@#$%^&*()<>{}\[\]|\/\\~`+-=]", text.lower())
            tokens = tokens[:self.max_len] + ['<PAD>'] * (self.max_len - len(tokens))
            return [self.vocabulary.get(token) for token in tokens]

        # Convert to tensors
        self.num_train = len(self.X_train)
        self.num_test = len(self.X_test)
        self.X_train = torch.tensor([tokenize_and_pad(text) for text in self.X_train], dtype=torch.long)
        self.y_train = torch.tensor(self.y_train, dtype=torch.float32)

        self.X_test = torch.tensor([tokenize_and_pad(text) for text in self.X_test], dtype=torch.long)
        self.y_test = torch.tensor(self.y_test, dtype=torch.float32)
        
        self.split = split

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        # xs_b = torch.randn(b_size, n_points, self.n_dims)
        if self.split == "train":
            indices = torch.randint(0, self.num_train, (b_size, n_points))
        else: 
            indices = torch.randint(0, self.num_test, (b_size, n_points))
        if self.split == "train":
            xs_b = self.X_train[indices]
            ys_b = self.y_train[indices]
        else:
            xs_b = self.X_test[indices]
            ys_b = self.y_test[indices]
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b, ys_b

class AGDataSampler(DataSampler):
    def __init__(self, n_dims: int, split: str = "train", bias: float = None, scale: float = None):
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale
        import json, re
        with open("src/reasoning_module/ag_news_vocabulary.json") as f:
            self.vocabulary = json.load(f) # dict mapping words to ids 
        self.vocabulary_size = len(self.vocabulary)
        print("vocab size", self.vocabulary_size)      
        self.max_len = 200  # max around 170 tokens
        # Load the dataset
        (self.X_train, self.y_train), (self.X_test, self.y_test) = load_data(
            data_path="~/deniselj/Desktop/tart-denise/datasets/", dataset="ag_news", input_key="text", seed=42
        )
        
        def tokenize_and_pad(text: str) -> list:
            # converts to one-hot encoded vector of length 8955
            tokens = re.findall(r"[\w']+|[.,!?;:\"\'@#$%^&*()<>{}\[\]|\/\\~`+-=]", text.lower())
            tokens = tokens[:self.max_len] + ['<PAD>'] * (self.max_len - len(tokens))
            return [self.vocabulary.get(token) for token in tokens]

        # Convert to tensors
        self.num_train = len(self.X_train)
        self.num_test = len(self.X_test)
        self.X_train = torch.tensor([tokenize_and_pad(text) for text in self.X_train], dtype=torch.long)
        self.y_train = torch.tensor(self.y_train, dtype=torch.float32)

        self.X_test = torch.tensor([tokenize_and_pad(text) for text in self.X_test], dtype=torch.long)
        self.y_test = torch.tensor(self.y_test, dtype=torch.float32)
        
        self.split = split

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        # xs_b = torch.randn(b_size, n_points, self.n_dims)
        if self.split == "train":
            indices = torch.randint(0, self.num_train, (b_size, n_points))
        else: 
            indices = torch.randint(0, self.num_test, (b_size, n_points))
        if self.split == "train":
            xs_b = self.X_train[indices]
            ys_b = self.y_train[indices]
        else:
            xs_b = self.X_test[indices]
            ys_b = self.y_test[indices]
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b, ys_b
