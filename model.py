import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Config, GPT2Model
from torch_geometric.nn import GATConv

device = "cuda" if torch.cuda.is_available() else "cpu"

### Models ###

class GraphNameEncoder(nn.Module):
    """
    Uses the GPT2 pretrained Embedding and a self attention layer
    to combine the name embeddings of varying sizes into the same size.
    """
    def __init__(self, gpt2_type="gpt2"):
        super(GraphNameEncoder, self).__init__()
        gpt2_model = GPT2Model.from_pretrained(gpt2_type)
        self.embedding = gpt2_model.wte
        self.attention = nn.MultiheadAttention(embed_dim=768, num_heads=4, batch_first=True)
        
    def forward(self, batch):
        names = batch.name
        x = batch.x.to(device)
        embedded_names = []
        # Unfortunately can't parallelize this since they're of varying sizes and gpt2 traditionally does not
        # use padding
        for b in range(len(names)):
            for v in range(len(names[b])):
                embedded = self.embedding(torch.Tensor(names[b][v]).int().to(device))
                attn_output, _ = self.attention(embedded.unsqueeze(0), embedded.unsqueeze(0), embedded.unsqueeze(0))
                embedded_names.append(attn_output.sum(dim=1))
        
        embedded_names = torch.stack(embedded_names).squeeze()
        x = torch.hstack([x, embedded_names])
        return x

class MovieGraphNetwork(nn.Module):
    """
    A modified Graph Attention Network.
    """
    def __init__(self, feat_dim, hidden_dim, heads, n_layers, num_classes):
        super(MovieGraphNetwork, self).__init__()

        self.name_encoder = GraphNameEncoder("gpt2")
        self.convs = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.n_layers = n_layers
        self.classifier = nn.Linear(hidden_dim, num_classes)

        a = nn.LeakyReLU()
        for i in range(n_layers):
            start_dim = hidden_dim if i else feat_dim
            conv = GATConv(start_dim, hidden_dim, heads=heads, edge_dim=1, concat=False)
            self.convs.append(conv)
            self.acts.append(a)

    def forward(self, batch):
        edge_index = batch.edge_index.to(device)
        edge_attr = batch.edge_attr.to(device)
        
        x = self.name_encoder(batch)
        
        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.acts[i](x)
        
        features = x
        x = self.classifier(features)
        return x, features
    
class MovieNetwork(nn.Module):
    """
    Just a basic Transformer Encoder with the name encoder.
    """
    def __init__(self, feat_dim, hidden_dim, heads, n_layers, num_classes):
        super(MovieNetwork, self).__init__()

        self.name_encoder = GraphNameEncoder("gpt2")
        self.n_layers = n_layers
        
        self.bottleneck = nn.Linear(feat_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(hidden_dim, heads, dim_feedforward=2*hidden_dim, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, batch):
        edge_index = batch.edge_index
        edge_index = edge_index.to(device)
        
        x = self.name_encoder(batch)
        x = self.bottleneck(x)

        features = self.encoder(x.unsqueeze(1))
        
        x = self.classifier(features)

        return x.squeeze()
        
def init_weights(m):
    """
    Initialize the weights of the network.
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)
