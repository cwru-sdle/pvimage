import numpy as np
import matplotlib.pyplot as plt
import math


def plot_image_grid(image_list, title_prefix="Image", main_title="Image Grid", 
                   figsize_per_image=(3, 3), max_cols=4, cmap='viridis'):
    """
    Plot a grid of images from a list of 2D arrays.
    
    Parameters:
    -----------
    image_list : list
        List of 2D numpy arrays (images)
    title_prefix : str
        Prefix for individual image titles
    main_title : str
        Main title for the entire figure
    figsize_per_image : tuple
        Size of each individual subplot
    max_cols : int
        Maximum number of columns in the grid
    cmap : str
        Colormap for displaying images
    """
    n_images = len(image_list)
    
    if n_images == 0:
        print("No images to display!")
        return
    
    # Calculate grid dimensions
    n_cols = min(max_cols, n_images)
    n_rows = math.ceil(n_images / n_cols)
    
    # Calculate figure size
    fig_width = n_cols * figsize_per_image[0]
    fig_height = n_rows * figsize_per_image[1]
    
    # Create the figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    fig.suptitle(main_title, fontsize=16, y=0.98)
    
    # Handle case where there's only one subplot
    if n_images == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot each image
    for i, img in enumerate(image_list):
        row = i // n_cols
        col = i % n_cols
        
        if n_rows == 1 and n_cols == 1:
            ax = axes[0]
        elif n_rows == 1:
            ax = axes[col]
        elif n_cols == 1:
            ax = axes[row]
        else:
            ax = axes[row, col]
        
        # Display the image
        im = ax.imshow(img, cmap=cmap, aspect='auto')
        ax.set_title(f'{title_prefix} {i+1}', fontsize=10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    # Hide any unused subplots
    total_subplots = n_rows * n_cols
    for i in range(n_images, total_subplots):
        row = i // n_cols
        col = i % n_cols
        
        if n_rows == 1 and n_cols == 1:
            continue
        elif n_rows == 1:
            axes[col].set_visible(False)
        elif n_cols == 1:
            axes[row].set_visible(False)
        else:
            axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def plot_summary_stats(image_list, title="Image Statistics"):
    """
    Plot summary statistics for the image list.
    """
    if not image_list:
        print("No images to analyze!")
        return
    
    stats = []
    for i, img in enumerate(image_list):
        stats.append({
            'index': i,
            'mean': np.mean(img),
            'std': np.std(img),
            'min': np.min(img),
            'max': np.max(img),
            'shape': img.shape
        })
    
    # Create summary plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=16)
    
    indices = [s['index'] for s in stats]
    means = [s['mean'] for s in stats]
    stds = [s['std'] for s in stats]
    mins = [s['min'] for s in stats]
    maxs = [s['max'] for s in stats]
    
    axes[0, 0].bar(indices, means)
    axes[0, 0].set_title('Mean Values')
    axes[0, 0].set_xlabel('Image Index')
    axes[0, 0].set_ylabel('Mean')
    
    axes[0, 1].bar(indices, stds)
    axes[0, 1].set_title('Standard Deviation')
    axes[0, 1].set_xlabel('Image Index')
    axes[0, 1].set_ylabel('Std Dev')
    
    axes[1, 0].bar(indices, mins, alpha=0.7, label='Min')
    axes[1, 0].bar(indices, maxs, alpha=0.7, label='Max')
    axes[1, 0].set_title('Min/Max Values')
    axes[1, 0].set_xlabel('Image Index')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].legend()
    
    # Shape information
    shapes = [f"{s['shape'][0]}x{s['shape'][1]}" for s in stats]
    axes[1, 1].text(0.1, 0.9, "Image Shapes:", transform=axes[1, 1].transAxes, fontweight='bold')
    for i, shape in enumerate(shapes):
        axes[1, 1].text(0.1, 0.8 - i*0.1, f"Image {i+1}: {shape}", 
                       transform=axes[1, 1].transAxes)
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
