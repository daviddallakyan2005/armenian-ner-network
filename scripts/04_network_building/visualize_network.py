import os
import argparse
import logging
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib
import numpy as np
import json

# Use Agg backend for environments without a display
matplotlib.use('Agg')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEFAULT_IMAGE_FORMAT = "png"
DEFAULT_OUTPUT_SUBDIR = "network_visualizations"

def get_project_root():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(script_dir))

def load_graph(file_path):
    """Loads a graph from a file (GEXF or GraphML)."""
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    graph = None
    try:
        if ext == '.gexf':
            graph = nx.read_gexf(file_path)
        elif ext == '.graphml':
            graph = nx.read_graphml(file_path)
        elif ext == '.json':
            from networkx.readwrite import json_graph
            with open(file_path, 'r', encoding='utf-8') as f:
                 data = json.load(f)
            graph = json_graph.node_link_graph(data)
        else:
            logging.warning(f"Unsupported graph file format: {ext} for file {file_path}. Skipping.")
            return None
        logging.info(f"Successfully loaded graph from {file_path} ({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)")
        return graph
    except Exception as e:
        logging.error(f"Failed to load graph from {file_path}: {e}")
        return None

def visualize_and_save_graph(graph, output_path, title="Character Network"):
    """Visualizes the graph using matplotlib and saves it to a file."""
    if graph is None or graph.number_of_nodes() == 0:
        logging.warning(f"Graph is empty or None, skipping visualization for {output_path}.")
        return False

    fig, ax = plt.subplots(figsize=(40, 40))
    ax.set_axis_off()

    try:
        logging.info("Attempting spring layout with k=2.0, iterations=150...")
        pos = nx.spring_layout(graph, k=2.0, iterations=150)
        logging.info("Spring layout successful.")
    except Exception as e:
        logging.warning(f"Spring layout failed ({e}), falling back to Kamada-Kawai layout.")
        try:
            pos = nx.kamada_kawai_layout(graph, scale=2)
        except Exception as e2:
            logging.error(f"Both spring and Kamada-Kawai layouts failed. Error: {e2}")
            pos = nx.random_layout(graph)

    node_degrees = dict(graph.degree())
    min_node_size = 150
    node_sizes = [(node_degrees.get(node, 0) * 20) + min_node_size for node in graph.nodes()]

    edges = graph.edges(data=True)
    edge_weights = np.array([d.get('weight', 1) for u, v, d in edges])
    
    # Prepare edge colors using viridis colormap
    if len(edge_weights) > 0:
        min_w, max_w = np.min(edge_weights), np.max(edge_weights)
        if max_w == min_w: # Handle constant weight case
            normalized_weights = np.ones_like(edge_weights) * 0.5
        else:
            normalized_weights = (edge_weights - min_w) / (max_w - min_w)
        
        cmap = cm.viridis 
        edge_colors = cmap(normalized_weights)
        
        min_width, max_width = 1.0, 7.0
        edge_widths = min_width + normalized_weights * (max_width - min_width)
        
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=min_w, vmax=max_w))
        sm.set_array([])

        # Add the colorbar to the figure, linked to the main axes
        cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
        cbar.set_label('Co-occurrence Frequency', rotation=270, labelpad=20, fontsize=12)

    else:
        edge_widths = []
        edge_colors = 'grey'
        min_w, max_w = 0, 0


    nx.draw_networkx_nodes(graph, pos, node_size=node_sizes, node_color='skyblue', alpha=0.8, ax=ax)
    nx.draw_networkx_edges(graph, pos, edgelist=list(graph.edges()), width=edge_widths, edge_color=edge_colors, alpha=0.7, ax=ax)
    nx.draw_networkx_labels(graph, pos, font_size=7, font_family='sans-serif', ax=ax) # Decreased font size from 9

    plt.title(title, fontsize=20)

    try:
        fig.savefig(output_path, format=output_path.split('.')[-1], bbox_inches='tight', dpi=300)
        logging.info(f"Graph visualization saved successfully to {output_path}")
        plt.close(fig)
        return True
    except Exception as e:
        logging.error(f"Failed to save visualization to {output_path}: {e}")
        plt.close(fig)
        return False


project_root = get_project_root()
default_input_dir = os.path.join(project_root, "results", "networks")
default_output_dir = os.path.join(project_root, "results", DEFAULT_OUTPUT_SUBDIR)

parser = argparse.ArgumentParser(description="Visualize character co-occurrence networks.")
parser.add_argument(
    "--input_dir",
    type=str,
    default=default_input_dir,
    help=f"Directory containing the network graph files (e.g., .gexf, .graphml) (default: {default_input_dir})"
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=default_output_dir,
    help=f"Directory to save the output visualization images (default: {default_output_dir})"
)
parser.add_argument(
    "--format",
    type=str,
    default=DEFAULT_IMAGE_FORMAT,
    choices=['png', 'jpg', 'svg', 'pdf'],
    help=f"Output image format (default: {DEFAULT_IMAGE_FORMAT})"
)
parser.add_argument(
    "--file",
    type=str,
    default=None,
    help="Visualize only a specific network graph file (relative to input_dir)."
)

args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

visualized_count = 0
if args.file:
    file_path = os.path.join(args.input_dir, args.file)
    if os.path.exists(file_path):
        graph = load_graph(file_path)
        if graph:
            base_name = os.path.splitext(args.file)[0]
            output_filename = f"{base_name}.{args.format}"
            output_path = os.path.join(args.output_dir, output_filename)
            if visualize_and_save_graph(graph, output_path, title=f"Network - {base_name}"):
                visualized_count = 1
    else:
        logging.error(f"Specified file not found: {file_path}")
else:
    try:
         graph_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(('.gexf', '.graphml', '.json'))]
         if not graph_files:
              logging.warning(f"No graph files (.gexf, .graphml, .json) found in {args.input_dir}")
    except FileNotFoundError:
         logging.error(f"Input directory not found: {args.input_dir}")
         graph_files = []

    for filename in graph_files:
        file_path = os.path.join(args.input_dir, filename)
        graph = load_graph(file_path)
        if graph:
            base_name = os.path.splitext(filename)[0]
            output_filename = f"{base_name}.{args.format}"
            output_path = os.path.join(args.output_dir, output_filename)
            if visualize_and_save_graph(graph, output_path, title=f"Network - {base_name}"):
                 visualized_count += 1

logging.info(f"--- Network Visualization Script Finished ---")
logging.info(f"Created {visualized_count} network visualizations in {args.output_dir}") 