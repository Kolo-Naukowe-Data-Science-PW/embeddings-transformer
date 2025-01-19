import torch
from transformer import MIDITransformerEncoder
from dataset import MIDIDataset
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px
import pandas as pd

def get_song_embedding(model, song_tokens, device):
    song_tensor = song_tokens.to(device)
    song_embedding = model.get_embeddings(song_tensor)[0]
    return song_embedding.cpu().numpy()

def prepare_embeddings(model, dataset, device):
    song_embeddings = []
    song_titles = []
    composers = []
    
    for i in range(len(dataset)):
        song = dataset[i]
        embedding = get_song_embedding(model, song, device)
        song_embeddings.append(embedding)
        
        info = dataset.data["info"][i]
        song_titles.append(info["title"])
        composers.append(info.get("composer", "Unknown"))
    
    embeddings_array = np.vstack(song_embeddings)
    
    # Wykonaj t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings_array)
    
    return embeddings_2d, song_titles, composers

def plot_interactive(embeddings_2d, song_titles, composers):
    """Creates interactive plot using Plotly with custom colors"""
    
    # Custom color sequence
    colors = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel + px.colors.qualitative.Bold
    
    df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'title': song_titles,
        'composer': composers
    })
    
    fig = px.scatter(
        df, x='x', y='y',
        hover_data=['title', 'composer'],
        color='composer',
        color_discrete_sequence=colors,
        title='Interactive Song Visualization (hover for details)'
    )
    
    fig.update_traces(marker=dict(
        size=10,
        line=dict(width=1, color='DarkSlateGrey')
    ))
    
    fig.update_layout(
        width=1200,
        height=800,
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(
            title="Composers",
            bordercolor="Black",
            borderwidth=1
        )
    )
    
    fig.write_html("interactive_embeddings_2048.html")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Załaduj dataset i model
    dataset = MIDIDataset(
        tokens_path="data/tokens.json",
        max_seq_len=1024
    )
    
    model = MIDITransformerEncoder(
        vocab_size=dataset.vocab_size,
        embed_dim=384,
        nhead=6,
        num_layers=4,
        max_seq_len=2048
    ).to(device)
    
    model.load_state_dict(torch.load("models/model_2048.pth")['model_state_dict'])
    model.eval()
    
    # Przygotuj dane
    embeddings_2d, song_titles, composers = prepare_embeddings(model, dataset, device)
    
    # Stwórz obie wizualizacje
    plot_interactive(embeddings_2d, song_titles, composers)

if __name__ == "__main__":
    main()