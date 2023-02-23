import torch
import torch.nn as nn


class NCF(nn.Module):
    def __init__(
            self,
            n_item: int,
            n_user: int,
            embedding_size: int,
            mlp_layer_dims: list,
            dropout_rate: float,
            use_gmf: bool,
    ):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.user_embedding = nn.Embedding(n_user, embedding_size)
        self.item_embedding = nn.Embedding(n_item, embedding_size)
        self.use_gmf = use_gmf

        self.layer_input_dim = embedding_size * 2
        mlp_layers = self.__set_layers(layer_dims=mlp_layer_dims)
        self.mlp_layers = nn.Sequential(*mlp_layers)

        if self.use_gmf:
            self.mf_user_embedding = nn.Embedding(n_user, embedding_size)
            self.mf_item_embedding = nn.Embedding(n_item, embedding_size)

            self.layer_input_dim += embedding_size

        self.predict_layer = nn.Linear(self.layer_input_dim, 1)

    def __set_layers(self, layer_dims):
        layers = list()
        for idx, layer_dim in enumerate(layer_dims, 1):
            layers.append(nn.Linear(self.layer_input_dim, layer_dim))
            if not idx == len(layer_dims):
                layers.append(nn.BatchNorm1d(layer_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=self.dropout_rate))
            self.layer_input_dim = layer_dim

        return layers

    def __init_weight_(self):
        nn.init.normal_(self.user_embedding.weight)
        nn.init.normal_(self.item_embedding.weight)

        for mlp_layer in self.mlp_layers:
            if isinstance(mlp_layer, nn.Linear):
                nn.init.kaiming_normal_(mlp_layer.weight, nonlinearity="relu")

        nn.init.kaiming_normal_(self.predict_layer.weight, nonlinearity="relu")

        if self.use_gmf:
            nn.init.normal_(self.mf_user_embedding.weight)
            nn.init.normal_(self.mf_item_embedding.weight)

    def forward(self, user_id, item_id):
        user_vector = self.user_embedding(user_id)
        item_vector = self.item_embedding(item_id)

        mlp_input = torch.cat((user_vector, item_vector), dim=-1)
        mlp_output = self.mlp_layers(mlp_input)

        if self.use_gmf:
            mf_user_vector = self.mf_user_embedding(user_id)
            mf_item_vector = self.mf_item_embedding(item_id)

            gmf_output = torch.mul(mf_user_vector, mf_item_vector)

            mlp_output = torch.cat((gmf_output, mlp_output), dim=-1)

        score = self.predict_layer(mlp_output)

        return score
