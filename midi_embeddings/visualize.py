import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import plotly.express as px
from sklearn.manifold import TSNE
from torch.utils.data import Dataset

from midi_embeddings.transformer import MIDITransformerEncoder
from midi_embeddings.dataset import MIDIDatasetDynamic, MIDIDatasetPresaved


def get_song_embedding(
    model: nn.Module,
    song_tokens: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Returns the embedding of a single song

    Args:
        model (nn.Module): model to use for embedding generation
        song_tokens (torch.Tensor): tokens of the song to be processed
        device (torch.device): device to use for processing

    Returns:
        np.ndarray: Given song's embedding.
    """

    song_tensor = song_tokens.to(device)
    song_embedding = model.get_embeddings(song_tensor)[0]  # returns [batch_size, embed_size]

    return song_embedding.cpu().numpy()


def prepare_embeddings(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
) -> tuple:
    """Generates embeddings for all songs in the dataset and performs t-SNE on them

    Args:
        model (nn.Module): Model to use for embedding generation.
        dataset (Dataset): Dataset containing the songs.
        device (torch.device): Device to perform the processing on.

    Returns:
        tuple: (embeddings array, song titles, composers)
    """
    song_embeddings = []
    song_titles = []
    composers = []

    # For each song: get embedding and store it with info
    for i in range(len(dataset)):
        song = dataset[i]["token_ids"]
        embedding = get_song_embedding(model, song, device)
        song_embeddings.append(embedding)

        info = dataset[i]["info"]

        song_titles.append(info["title"])
        composers.append(info.get("composer", "Unknown"))

    # Stack all embeddings into a single array
    embeddings_array = np.vstack(song_embeddings)

    # Perform t-SNE on the data
    tsne = TSNE(random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings_array)

    return embeddings_2d, song_titles, composers


def plot_interactive(
    embeddings_2d: np.ndarray,
    song_titles: list[str],
    composers: list[str],
    file_name: str,
) -> None:
    """Creates interactive plot using Plotly with custom colors

    Args:
        embeddings_2d (np.ndarray): 2D embeddings of the songs.
        song_titles (list[str]): Titles of the songs.
        composers (list[str]): Composers of the songs.
    """

    colors = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel + px.colors.qualitative.Bold

    df = pd.DataFrame(
        {"x": embeddings_2d[:, 0], "y": embeddings_2d[:, 1], "title": song_titles, "composer": composers},
    )

    fig = px.scatter(
        df,
        x="x",
        y="y",
        hover_data=["title", "composer"],
        color="composer",
        color_discrete_sequence=colors,
        title="Interactive Song Visualization (hover for details)",
    )

    fig.update_traces(marker=dict(size=10, line=dict(width=1, color="DarkSlateGrey")))

    fig.update_layout(
        width=1200,
        height=800,
        showlegend=True,
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(title="Composers", bordercolor="Black", borderwidth=1),
    )

    fig.write_html(file_name)


def visualize_embeddings(
    model: nn.Module,
    device: torch.device,
    max_seq_len: int = 2048,
    limit: int = 100,
    file_name: str = "embeddings.html",
):
    """Simple interface for visualizing song embeddings

    Args:
        model: Trained model to use for embeddings
        device: Device to use for computation
        max_seq_len: Sequence length for processing
        limit: Number of songs to visualize (for faster testing)
        file_name: Name of the output file
    """
    # Load dataset with metadata
    dataset = MIDIDatasetPresaved(
        split="all", max_seq_len=max_seq_len, tokenizer_path="awesome.json", return_info=True, limit=limit
    )

    # Generate and plot embeddings
    embeddings_2d, titles, composers = prepare_embeddings(model, dataset, device)
    plot_interactive(embeddings_2d, titles, composers, file_name)


def main():
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model checkpoint (state_dict and config)
    checkpoint = torch.load("models/model_2048.pth")

    # Load dataset and model
    dataset = MIDIDatasetDynamic(
        split="all",
        max_seq_len=checkpoint["config"]["max_seq_len"],
        return_info=True,  # to get titles and composers
        limit=100,
        tokenizer_path="awesome.json",
    )

    model = MIDITransformerEncoder(
        vocab_size=dataset.vocab_size,
        embed_dim=384,
        nhead=6,
        num_layers=4,
        max_seq_len=2048,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Prepare the data
    embeddings_2d, song_titles, composers = prepare_embeddings(model, dataset, device)

    # Create and save visualization
    plot_interactive(embeddings_2d, song_titles, composers)


if __name__ == "__main__":
    main()
