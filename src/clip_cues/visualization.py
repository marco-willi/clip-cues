"""Visualization utilities for CLIP-Cues."""

import math
from textwrap import wrap

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from torchvision.transforms.v2 import functional as TF


def plot_collage(
    images: list[torch.Tensor | Image.Image],
    captions: list[str] | None = None,
    row_labels: list[str] | None = None,
    row_labels_position: str = "left",  # 'left' or 'right'
    row_labels_font_size: int | None = None,  # If None, defaults to caption_font_size
    col_labels: list[str] | None = None,
    col_labels_position: str = "top",  # 'top' or 'bottom'
    col_labels_font_size: int | None = None,  # If None, defaults to caption_font_size
    nrows: int | None = None,
    ncols: int | None = None,
    caption_width: int | None = 30,
    caption_font_size: int = 10,
    global_normalize: bool = False,
    figsize: tuple[int, int] | None = None,
    title: str | None = None,
    sub_title: str | None = None,
    axes_iteration_order: str = "C",  # 'C' for C-style (row-major), 'F' for Fortran-style (column-major)
    aspect: str = "auto",  # 'auto' or 'equal' - controls image aspect ratio in subplots
):
    """
    Plot a collage of images with optional captions and row labels in a grid of specified rows and columns.

    Args:
        images: List of images (torch.Tensor or PIL.Image)
        captions: Optional captions for each image
        row_labels: Optional labels for each row
        row_labels_position: Position of row labels ('left' or 'right')
        row_labels_font_size: Font size for row labels
        col_labels: Optional labels for each column
        col_labels_position: Position of column labels ('top' or 'bottom')
        col_labels_font_size: Font size for column labels
        nrows: Number of rows (auto-calculated if None)
        ncols: Number of columns (auto-calculated if None)
        caption_width: Maximum width for captions (wraps text)
        caption_font_size: Font size for captions
        global_normalize: Whether to normalize across all images
        figsize: Figure size tuple (width, height)
        title: Main title for the figure
        sub_title: Subtitle for the figure
        axes_iteration_order: Order to iterate through axes ('C' or 'F')
        aspect: Controls how images are displayed in subplots
            - 'auto' (default): Images stretch to fill subplot (may distort)
            - 'equal': Preserves image aspect ratio (may leave whitespace)

    Returns:
        fig: Matplotlib figure
        axes: Matplotlib axes array

    Example:
        >>> from PIL import Image
        >>> images = [Image.open(f"img{i}.jpg") for i in range(9)]
        >>> fig, axes = plot_collage(
        ...     images,
        ...     nrows=3,
        ...     ncols=3,
        ...     title="Image Gallery"
        ... )
        >>> plt.show()
    """

    num_images = len(images)

    # Default col_labels_font_size and row_labels_font_size to caption_font_size if not specified
    if col_labels_font_size is None:
        col_labels_font_size = caption_font_size
    if row_labels_font_size is None:
        row_labels_font_size = caption_font_size

    # If nrows and ncols are not specified, create a square grid
    if nrows is None or ncols is None:
        nrows = ncols = math.ceil(math.sqrt(num_images))

    # Convert tensors to PIL images if needed
    pil_images = [TF.to_pil_image(img) if isinstance(img, torch.Tensor) else img for img in images]

    if global_normalize:
        # Find global min and max across all PIL images for normalization
        all_images = np.concatenate([np.array(img).flatten() for img in pil_images])
        vmin, vmax = all_images.min(), all_images.max()
    else:
        vmin, vmax = 0.0, 255.0

    if figsize is None:
        figsize = (ncols * 2, nrows * 2)

    # Create the figure for the collage
    fig, axes = plt.subplots(
        figsize=figsize,
        nrows=nrows,
        ncols=ncols,
    )

    if title:
        fig.suptitle(title, fontsize=20)

    if sub_title:
        fig.text(0.5, 0.95, sub_title, horizontalalignment="center")

    # Flatten axes for easier iteration
    axes1d = axes.flatten(order=axes_iteration_order)

    for i, ax in enumerate(axes1d):
        if i < num_images:
            ax.imshow(np.array(pil_images[i]), vmin=vmin, vmax=vmax, aspect=aspect)

    # Set Captions
    if captions is not None:
        for i, ax in enumerate(axes1d):
            if i < len(captions):
                caption = captions[i]
                if caption_width is not None:
                    caption = "\n".join(wrap(caption, caption_width))
                # If col_labels are provided, captions go to the opposite position
                # Otherwise, captions use col_labels_position
                if col_labels is not None:
                    # Put captions opposite to col_labels
                    caption_pos = "bottom" if col_labels_position == "top" else "top"
                else:
                    caption_pos = col_labels_position

                if caption_pos == "top":
                    ax.set_title(caption, fontsize=caption_font_size)
                elif caption_pos == "bottom":
                    ax.set_xlabel(caption, fontsize=caption_font_size)

    # Format axes
    for i, ax in enumerate(axes1d):
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.yaxis.set_ticks([])
        ax.xaxis.set_ticks([])
        if i >= num_images:
            ax.axis("off")

    # Add column labels at the top or bottom based on col_labels_position
    if col_labels is not None:
        for col_idx, label in enumerate(col_labels):
            if col_idx < ncols:
                # Handle both 1D (single row) and 2D axes arrays
                if col_labels_position == "top":
                    if nrows == 1:
                        axes[col_idx].set_title(label, fontsize=col_labels_font_size, pad=20)
                    else:
                        axes[0, col_idx].set_title(label, fontsize=col_labels_font_size, pad=20)
                elif col_labels_position == "bottom":
                    if nrows == 1:
                        axes[col_idx].set_xlabel(label, fontsize=col_labels_font_size)
                    else:
                        axes[-1, col_idx].set_xlabel(label, fontsize=col_labels_font_size)

    # Add row labels at the left or right of each row based on row_labels_position
    if row_labels is not None:
        for row_idx, label in enumerate(row_labels):
            if row_idx < nrows:
                # Determine which column to use and which y-axis based on position
                if row_labels_position == "left":
                    col_idx = 0  # First column (left side)
                    # Use left y-axis (default)
                    ylabel_side = "left"
                else:  # "right"
                    # For right position, find the last image in this row
                    # Calculate the last image index in this row
                    row_start = row_idx * ncols
                    row_end = min(row_start + ncols, num_images)
                    # If this row has images, use the last one; otherwise use rightmost column
                    if row_end > row_start:
                        col_idx = (row_end - 1) % ncols  # Column index of last image in row
                    else:
                        col_idx = ncols - 1  # Default to rightmost column
                    # Use right y-axis to place label on the right side
                    ylabel_side = "right"

                # Handle both 1D (single row/column) and 2D axes arrays
                if nrows == 1:
                    ax = axes[col_idx]
                elif ncols == 1:
                    ax = axes[row_idx]
                else:
                    ax = axes[row_idx, col_idx]

                # Place label on appropriate y-axis
                if ylabel_side == "left":
                    ax.set_ylabel(label, rotation=90, fontsize=row_labels_font_size, labelpad=5)
                else:  # "right"
                    # Use the right y-axis
                    ax.yaxis.set_label_position("right")
                    ax.set_ylabel(label, rotation=270, fontsize=row_labels_font_size, labelpad=5)

    return fig, axes
