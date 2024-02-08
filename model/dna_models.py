import copy

from torch import nn
import torch.nn.functional as F
from model.promoter_model import GaussianFourierProjection, Dense


class MLPModel(nn.Module):
    def __init__(self, args, alphabet_size, num_cls, classifier=False):
        super().__init__()
        self.alphabet_size = alphabet_size
        self.classifier = classifier
        self.num_cls = num_cls


        self.time_embedder = nn.Sequential(GaussianFourierProjection(embed_dim= args.hidden_dim),nn.Linear(args.hidden_dim, args.hidden_dim))
        self.embedder = nn.Linear((1 if classifier and not args.cls_expanded_simplex else 2) * self.alphabet_size,  args.hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim if classifier else self.alphabet_size)
        )
        if classifier:
            self.cls_head = nn.Sequential(nn.Linear(args.hidden_dim, args.hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(args.hidden_dim, self.num_cls))
        if self.args.cls_free_guidance and not self.classifier:
            self.cls_embedder = nn.Embedding(num_embeddings=self.num_cls + 1, embedding_dim=args.hidden_dim)


    def forward(self, seq,t, cls=None):
        time_embed = self.time_embedder(t)
        feat = self.embedder(seq)
        feat = feat + time_embed[:,None,:]
        if self.args.cls_free_guidance and not self.classifier:
            feat = feat + self.cls_embedder(cls)[:, None, :]
        feat = self.mlp(feat)
        if self.classifier:
            return self.cls_head(feat.mean(dim=1))
        else:
            return feat

class CNNModel(nn.Module):
    def __init__(self, args, alphabet_size, num_cls, classifier=False):
        super().__init__()
        self.alphabet_size = alphabet_size
        self.args = args
        self.classifier = classifier
        self.num_cls = num_cls

        if self.args.clean_data:
            self.linear = nn.Embedding(self.alphabet_size, embedding_dim=args.hidden_dim)
        else:
            expanded_simplex_input = args.cls_expanded_simplex or not classifier and (args.mode == 'dirichlet' or args.mode == 'riemannian')
            inp_size = self.alphabet_size * (2 if expanded_simplex_input else 1)
            if (args.mode == 'ardm' or args.mode == 'lrar') and not classifier:
                inp_size += 1 # plus one for the mask token of these models
            self.linear = nn.Conv1d(inp_size, args.hidden_dim, kernel_size=9, padding=4)
            self.time_embedder = nn.Sequential(GaussianFourierProjection(embed_dim= args.hidden_dim),nn.Linear(args.hidden_dim, args.hidden_dim))

        self.num_layers = 5 * args.num_cnn_stacks
        self.convs = [nn.Conv1d(args.hidden_dim, args.hidden_dim, kernel_size=9, padding=4),
                                     nn.Conv1d(args.hidden_dim, args.hidden_dim, kernel_size=9, padding=4),
                                     nn.Conv1d(args.hidden_dim, args.hidden_dim, kernel_size=9, dilation=4, padding=16),
                                     nn.Conv1d(args.hidden_dim, args.hidden_dim, kernel_size=9, dilation=16, padding=64),
                                     nn.Conv1d(args.hidden_dim, args.hidden_dim, kernel_size=9, dilation=64, padding=256)]
        self.convs = nn.ModuleList([copy.deepcopy(layer) for layer in self.convs for i in range(args.num_cnn_stacks)])
        self.time_layers = nn.ModuleList([Dense(args.hidden_dim, args.hidden_dim) for _ in range(self.num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(args.hidden_dim) for _ in range(self.num_layers)])
        self.final_conv = nn.Sequential(nn.Conv1d(args.hidden_dim, args.hidden_dim, kernel_size=1),
                                   nn.ReLU(),
                                   nn.Conv1d(args.hidden_dim, args.hidden_dim if classifier else self.alphabet_size, kernel_size=1))
        self.dropout = nn.Dropout(args.dropout)
        if classifier:
            self.cls_head = nn.Sequential(nn.Linear(args.hidden_dim, args.hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(args.hidden_dim, self.num_cls))

        if self.args.cls_free_guidance and not self.classifier:
            self.cls_embedder = nn.Embedding(num_embeddings=self.num_cls + 1, embedding_dim=args.hidden_dim)
            self.cls_layers = nn.ModuleList([Dense(args.hidden_dim, args.hidden_dim) for _ in range(self.num_layers)])
    def forward(self, seq, t, cls = None, return_embedding=False):
        if self.args.clean_data:
            feat = self.linear(seq)
            feat = feat.permute(0, 2, 1)
        else:
            time_emb = F.relu(self.time_embedder(t))
            feat = seq.permute(0, 2, 1)
            feat = F.relu(self.linear(feat))

        if self.args.cls_free_guidance and not self.classifier:
            cls_emb = self.cls_embedder(cls)

        for i in range(self.num_layers):
            h = self.dropout(feat.clone())
            if not self.args.clean_data:
                h = h + self.time_layers[i](time_emb)[:, :, None]
            if self.args.cls_free_guidance and not self.classifier:
                h = h + self.cls_layers[i](cls_emb)[:, :, None]
            h = self.norms[i]((h).permute(0, 2, 1))
            h = F.relu(self.convs[i](h.permute(0, 2, 1)))
            if h.shape == feat.shape:
                feat = h + feat
            else:
                feat = h
        feat = self.final_conv(feat)
        feat = feat.permute(0, 2, 1)
        if self.classifier:
            feat = feat.mean(dim=1)
            if return_embedding:
                embedding = self.cls_head[:1](feat)
                return self.cls_head[1:](embedding), embedding
            else:
                return self.cls_head(feat)
        return feat

class TransformerModel(nn.Module):
    def __init__(self, args, alphabet_size, num_cls, classifier=False):
        super().__init__()
        self.alphabet_size = alphabet_size
        self.classifier = classifier
        self.num_cls = num_cls
        self.args = args
        if self.args.clean_data:
            self.embedder = nn.Embedding(self.alphabet_size, args.hidden_dim)
        else:
            expanded_simplex_input = args.cls_expanded_simplex or not classifier and (args.mode == 'dirichlet' or args.mode == 'riemannian')
            self.embedder = nn.Linear((2 if expanded_simplex_input  else 1) *  self.alphabet_size,  args.hidden_dim)
            self.time_embedder = nn.Sequential(GaussianFourierProjection(embed_dim=args.hidden_dim), nn.Linear(args.hidden_dim, args.hidden_dim))
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=args.hidden_dim, nhead=4, dim_feedforward=args.hidden_dim, dropout=args.dropout), num_layers=args.num_layers, norm=nn.LayerNorm(args.hidden_dim))

        if self.classifier:
            self.cls_head = nn.Sequential(nn.Linear(args.hidden_dim, args.hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(args.hidden_dim, self.num_cls))
        else:
            self.out = nn.Linear(args.hidden_dim, self.alphabet_size)

        if self.args.cls_free_guidance and not self.classifier:
            self.cls_embedder = nn.Embedding(num_embeddings=self.num_cls + 1, embedding_dim=args.hidden_dim)

    def forward(self, seq, t, cls = None):
        feat = self.embedder(seq)
        if not self.args.clean_data:
            time_embed = F.relu(self.time_embedder(t))
            feat = feat + time_embed[:,None,:]
        if self.args.cls_free_guidance and not self.classifier:
            feat = feat + self.cls_embedder(cls)[:,None,:]
        feat = self.transformer(feat)
        if self.classifier:
            feat = feat.mean(dim=1)
            return self.cls_head(feat)
        else:
            return self.out(feat)


class DeepFlyBrainModel(nn.Module):
    def __init__(self, args, alphabet_size, num_cls, classifier=False):
        super(DeepFlyBrainModel, self).__init__()
        self.alphabet_size = alphabet_size
        self.classifier = classifier
        self.num_cls = num_cls
        self.args = args
        self.embedder = nn.Embedding(self.alphabet_size, args.hidden_dim)
        self.conv1d = nn.Conv1d(in_channels=args.hidden_dim, out_channels=1024, kernel_size=24, stride=1, padding='same')
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=12, stride=12)
        self.dropout1 = nn.Dropout(0.5)
        # TimeDistributed layer with Dense - implemented using Conv1d with kernel size 1
        self.time_distributed = nn.Conv1d(in_channels=1024, out_channels=128, kernel_size=1)
        # Bidirectional LSTM
        self.bidirectional = nn.LSTM(input_size=128, hidden_size=128, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(0.5)
        self.dense1 = nn.Linear(in_features=128*2*41, out_features=256)  # Assuming output from flatten is (batch_size, 128*2*42)
        self.dropout3 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(in_features=256, out_features=self.num_cls)

    def forward(self, seq, t, cls = None):
        x = self.embedder(seq)
        x = x.permute(0, 2, 1)  # Adjusting for Conv1D input
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout1(x)
        x = self.time_distributed(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)  # Adjusting for LSTM input
        x, _ = self.bidirectional(x)
        x = self.dropout2(x)
        x = x.reshape(seq.shape[0], -1)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout3(x)
        x = self.dense2(x)
        return x

