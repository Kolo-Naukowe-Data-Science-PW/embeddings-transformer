"""
Visualization Module (visualize.py)
-----------------------------------
This module provides tools for visualizing MIDI song embeddings using
Plotly and t-SNE. It includes:

- `get_song_embedding`: Extracts embeddings from a trained model.
- `prepare_embeddings`: Computes embeddings and reduces dimensions using t-SNE.
- `plot_interactive`: Creates an interactive scatter plot.
- `visualize_embeddings`: A utility function to load data, compute embeddings
  and visualize them.
"""


import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import plotly.express as px
from sklearn.manifold import TSNE
from torch.utils.data import Dataset, DataLoader

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


# def prepare_embeddings(
#     model: nn.Module,
#     dataset: Dataset,
#     device: torch.device,
# ) -> tuple:
#     """Generates embeddings for all songs in the dataset and performs t-SNE on them

#     Args:
#         model (nn.Module): Model to use for embedding generation.
#         dataset (Dataset): Dataset containing the songs.
#         device (torch.device): Device to perform the processing on.

#     Returns:
#         tuple: (embeddings array, song titles, composers)
#     """
#     song_embeddings = []
#     song_titles = []
#     composers = []

#     # For each song: get embedding and store it with info
#     for i in range(len(dataset)):
#         song_tokens = dataset[i]["token_ids"]
#         song_tokens = song_tokens.unsqueeze(0)  # add batch dimension
#         embedding = get_song_embedding(
#             model=model,
#             song_tokens=song_tokens,
#             device=device,
#         )
#         song_embeddings.append(embedding)

#         info = dataset[i]["info"]

#         song_titles.append(info["title"])
#         composers.append(info.get("composer", "Unknown"))

#     # Stack all embeddings into a single array
#     embeddings_array = np.vstack(song_embeddings)

#     # Perform t-SNE on the data
#     tsne = TSNE(random_state=42)
#     embeddings_2d = tsne.fit_transform(embeddings_array)

#     return embeddings_2d, song_titles, composers


def prepare_batched_embeddings(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 16,
    reference_embeddings: np.ndarray = None,
    align: bool = False,
) -> tuple:
    """Generates embeddings for all songs in the dataset and performs t-SNE on them

    Args:
        model (nn.Module): Model to use for embedding generation.
        dataset (Dataset): Dataset containing the songs.
        device (torch.device): Device to perform the processing on.
        batch_size (int, optional): Batch size for processing. Defaults to 16.
        reference_embeddings (np.ndarray, optional): Reference embedding for alignment. Defaults to None.
        align (bool, optional): Whether to align the embeddings. Defaults to False.

    Returns:
        tuple: (embeddings_2d array, raw_embeddings array, song_titles, composers)
    """

    # Get embeddings for all songs
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    song_embeddings = []

    for batch in dataloader:
        song_tokens = batch["token_ids"].to(device)
        embeddings = model.get_embeddings(song_tokens)
        song_embeddings.append(embeddings.cpu().numpy())

    raw_embeddings = np.vstack(song_embeddings)

    tsne = TSNE(random_state=42)
    embeddings_2d = tsne.fit_transform(raw_embeddings)

    song_titles = []
    composers = []

    # For each song: get embedding and store it with info
    for i in range(len(dataset)):
        info = dataset[i]["info"]
        song_titles.append(info["title"])
        composers.append(info.get("composer", "Unknown"))

    return embeddings_2d, song_titles, composers


def plot_interactive(
    embeddings_2d: np.ndarray,
    song_titles: list[str],
    composers: list[str],
    file_name: str = None,
    return_figure: bool = False,
) -> None:
    """Creates interactive plot using Plotly with custom colors

    Args:
        embeddings_2d (np.ndarray): 2D embeddings of the songs.
        song_titles (list[str]): Titles of the songs.
        composers (list[str]): Composers of the songs.
        file_name (str): Name of the output file.
        return_figure (bool): Whether to return the Plotly figure.
    """

    colors = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel + px.colors.qualitative.Bold

    df = pd.DataFrame(
        {
            "x": embeddings_2d[:, 0],
            "y": embeddings_2d[:, 1],
            "title": song_titles,
            "composer": composers,
        }
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

    if file_name:
        fig.write_html(file_name)
    if return_figure:
        return fig
    return None


def visualize_embeddings(
    model: nn.Module,
    device: torch.device,
    max_seq_len: int = 2048,
    limit: int = None,
    file_name: str = "embeddings.html",
    dataset: Dataset = None,
):
    """Simple interface for visualizing song embeddings

    Args:
        model: Trained model to use for embeddings
        device: Device to use for computation
        max_seq_len: Sequence length for processing
        limit: Number of songs to visualize (for faster testing)
        file_name: Name of the output file
        dataset: Dataset to use for visualization
    """
    # Load dataset with metadata
    if dataset is None:
        dataset = MIDIDatasetPresaved(
            split="all",
            max_seq_len=max_seq_len,
            tokenizer_path="awesome.json",
            return_info=True,
            limit=limit,
        )

    # Generate and plot embeddings
    embeddings_2d, titles, composers = prepare_batched_embeddings(
        model=model,
        dataset=dataset,
        device=device,
    )

    plot_interactive(
        embeddings_2d=embeddings_2d,
        song_titles=titles,
        composers=composers,
        file_name=file_name,
    )


class EmbeddingVisualizer:
    """Handles embedding visualization with tSNE alignment and animation"""

    def __init__(
        self,
        model: nn.Module,
        dataset: Dataset,
        device: torch.device,
        viz_interval: int = 2,
    ):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.embeddings_history = []
        self.titles = []
        self.composers = []
        self.viz_interval = viz_interval

    def _calculate_embeddings(self) -> tuple:
        """Calculate raw embeddings and reduce dimensions"""
        dataloader = DataLoader(self.dataset, batch_size=16, shuffle=False)
        all_embeddings = []

        # Get embeddings
        for batch in dataloader:
            song_tokens = batch["token_ids"].to(self.device)
            embeddings = self.model.get_embeddings(song_tokens)
            all_embeddings.append(embeddings.cpu().numpy())

        # Process metadata
        titles = []
        composers = []
        for i in range(len(self.dataset)):
            info = self.dataset[i]["info"]
            titles.append(info["title"])
            composers.append(info.get("composer", "Unknown"))

        raw_embeddings = np.vstack(all_embeddings)

        tsne = TSNE(random_state=42)
        embeddings_2d = tsne.fit_transform(raw_embeddings)

        return embeddings_2d, titles, composers

    def create_animation(self, file_name: str = "embedding_evolution.html") -> None:
        """Create animated plot of embedding evolution

        Args:
            file_name: Name of the output file
        """
        if not self.embeddings_history:
            raise ValueError("No embeddings history to animate")

        df = pd.DataFrame()
        for epoch, emb in enumerate(self.embeddings_history):
            epoch = epoch * self.viz_interval

            epoch_df = pd.DataFrame(
                {
                    "x": emb[:, 0],
                    "y": emb[:, 1],
                    "title": self.titles,
                    "composer": self.composers,
                    "epoch": [epoch + 1] * len(emb),
                }
            )
            df = pd.concat([df, epoch_df])

        fig = px.scatter(
            df,
            x="x",
            y="y",
            animation_frame="epoch",
            hover_data=["title", "composer"],
            color="composer",
            title="Embedding Evolution",
            color_discrete_sequence=px.colors.qualitative.Set3,
        )

        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 2000
        fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 500

        fig.write_html(file_name)

    def log_embeddings(self, epoch: int) -> dict:
        """Log embeddings for current epoch (to be called from training loop)

        Args:
            epoch: Current epoch number
        """
        if (epoch % self.viz_interval == 0) or (epoch == 1):
            embeddings_2d, self.titles, self.composers = self._calculate_embeddings()
            self.embeddings_history.append(embeddings_2d)

            fig = self.plot_current_embeddings()
            return {"embeddings_plot": fig, "embeddings_data": embeddings_2d}
        return {}

    def plot_current_embeddings(self) -> px.scatter:
        """Plot current epoch's embeddings"""
        if not self.embeddings_history:
            embeddings_2d, self.titles, self.composers = self._calculate_embeddings()
            self.embeddings_history.append(embeddings_2d)

        current_embeddings = self.embeddings_history[-1]

        return plot_interactive(
            embeddings_2d=current_embeddings, song_titles=self.titles, composers=self.composers, return_figure=True
        )


# Main function to run the visualization manually
def main():
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model checkpoint (state_dict and config)
    checkpoint = torch.load("models/model_best.pth")

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
        num_layers=6,
        max_seq_len=2048,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Prepare the data
    embeddings_2d, song_titles, composers = prepare_batched_embeddings(
        model=model,
        dataset=dataset,
        device=device,
        batch_size=32,
    )

    # Create and save visualization
    plot_interactive(
        embeddings_2d=embeddings_2d,
        song_titles=song_titles,
        composers=composers,
        file_name="test.html",
    )


if __name__ == "__main__":
    main()
