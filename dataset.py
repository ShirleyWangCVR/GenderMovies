import ast
import networkx as nx
import pandas as pd
from transformers import GPT2Tokenizer
from torch_geometric.data import Dataset
from torch_geometric.utils import from_networkx
from tqdm import trange, tqdm

class MovieGalaxiesDataset(Dataset):
    """
    Dataset for MovieGalaxies that also aggregates additional metadata
    """
    
    def __init__(self, root, split='train', transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        
        # I'm the only one going to be using these so I didn't bother making it generalizable
        # load
        credits = pd.read_csv("data/MoviesDataset/credits.csv")
        metadata = pd.read_csv("data/MoviesDataset/movies_metadata.csv")
        links = pd.read_csv("data/MoviesDataset/links.csv")
        bechedel_data = pd.read_json("data/bechedeltest.json", orient='records')
        galaxies_metadata = pd.read_table("data/MovieGalaxies/network_metadata.tab")

        # cleaning
        movie_data = pd.merge(credits, metadata, on='id', how='inner')
        movie_data = movie_data[~pd.isnull(movie_data['imdb_id'])]
        movie_data['imdb_id'] = movie_data['imdb_id'].apply(lambda x: x[2:])
        galaxies_metadata['IMDB_id'] = galaxies_metadata['IMDB_id'].apply(lambda x: x[2:])
        
        # add bechedel data info
        #movie_data = pd.merge(movie_data, bechedel_data, left_on='imdb_id', right_on='imdbid', how='inner')
        
        # split those with metadata and those without
        all_galaxies = pd.merge(galaxies_metadata, movie_data, how='left', left_on='IMDB_id', right_on='imdb_id')
        all_galaxies_with_meta = all_galaxies[~pd.isna(all_galaxies['imdb_id'])]
        all_galaxies_no_meta = all_galaxies[pd.isna(all_galaxies['imdb_id'])]
        all_galaxies_no_meta = pd.merge(galaxies_metadata, all_galaxies_no_meta[['IMDB_id']], how='inner', on='IMDB_id')
        
        if split == 'train':
            meta_dataframe = all_galaxies
        elif split == 'val':
            meta_dataframe = all_galaxies_no_meta
        
        # Load GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        
        self.networks = []
        for gid in tqdm(range(len(meta_dataframe))):
            filename = "data/MovieGalaxies/gexf/" + str(meta_dataframe.iloc[gid]['GexfID']) + ".gexf"
            graph = nx.read_gexf(filename)

            # get credits
            credits_str = meta_dataframe.iloc[gid]['cast']
            if pd.isna(credits_str):
                credits_cast = []
            else:
                credits_cast = ast.literal_eval(meta_dataframe.iloc[gid]['cast'])
            
            graph = self._add_gender(graph, credits_cast)
            graph = self._cleanup_graph(graph, tokenizer)
            
            torch_graph = from_networkx(graph)
            torch_graph.num_nodes = len(torch_graph.x)
            torch_graph.gid = meta_dataframe.iloc[gid]['GexfID']
            self.networks.append(torch_graph)
        
    def _add_gender(self, g, credit_cast):
        """
        Adds gender to the graph using the credits_cast list.
        1 = Male, 0 = Female, -1 = Unknown
        """
        num_found = 0
        for nodeidx in g.nodes:
            curr_node = g.nodes[nodeidx]
            name = curr_node['label'].lower()

            found = False
            for x in credit_cast:
                character_name = x['character'].lower()
                if name in character_name:
                    found = True
                    g.nodes[nodeidx]['gender'] = x['gender'] - 1
                    break
                elif name.startswith('miss ') or name.startswith('mr ') or name.startswith('ms ') or name.startswith('mrs '):
                    if len(character_name.split(' ')) > 1 and name.split(' ')[1] in character_name.split(' ')[1]:
                        found = True
                        g.nodes[nodeidx]['gender'] = x['gender'] - 1
                        break

            if not found:
                g.nodes[nodeidx]['gender'] = -1
            else:
                num_found += 1

        #print(str(round((num_found / len(g.nodes)) * 100, 2)) + "% Found")
        return g
    
    def _cleanup_graph(self, graph, text_encoder):
        graph.graph.pop('mode')
        graph.graph.pop('node_default')
        graph.graph.pop('edge_default')
        
        for v in graph.nodes:
            graph.nodes[v]['size'] = graph.nodes[v]['viz']['size']
            graph.nodes[v].pop('viz')
            graph.nodes[v].pop('movie_id')

            # encode name
            graph.nodes[v]['name'] = text_encoder.encode(graph.nodes[v]['label'].lower())
            graph.nodes[v].pop('label')
            
            x = []
            for key in graph.nodes[v].keys():
                if key != 'name' and key != 'gender':
                    x.append(graph.nodes[v][key])
                elif key == 'gender':
                    y = graph.nodes[v][key]

            graph.nodes[v]['x'] = x
            graph.nodes[v]['y'] = y
                

        for e in graph.edges:
            graph.edges[e].pop('label')
            graph.edges[e].pop('movie_id')
            graph.edges[e]['edge_attr'] = graph.edges[e]['weight']
            
        return graph
    
    def len(self):
        return len(self.networks)
    
    def get(self, idx):
        self.networks[idx].name = [list(x) for x in self.networks[idx].name]
        return self.networks[idx]
