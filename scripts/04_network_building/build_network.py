import os
import json
import argparse
import logging
from collections import defaultdict
from itertools import combinations
import networkx as nx


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEFAULT_ENTITY_LABEL = "PER"
DEFAULT_OUTPUT_EXTENSION = "gexf"


def get_project_root():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Assumes script is in <root>/scripts/04_network_building
    return os.path.dirname(os.path.dirname(script_dir))

def extract_characters(entity_list, label=DEFAULT_ENTITY_LABEL):
    """Extracts unique character names from a list of NER entities for a segment."""
    characters = set()
    if not isinstance(entity_list, list):
        logging.warning(f"Expected a list of entities, got {type(entity_list)}. Skipping segment.")
        return characters

    for entity in entity_list:
        # Check if entity is a dictionary and has the required keys
        if isinstance(entity, dict) and 'entity_group' in entity and 'word' in entity:
            if entity['entity_group'] == label:
                character_name = entity['word'].strip()
                if character_name: # Avoid adding empty strings
                    characters.add(character_name)
        elif isinstance(entity, dict) and 'error' in entity:
             logging.warning(f"Found error marker in entity list: {entity['error']}")
        else:
            logging.debug(f"Skipping malformed entity: {entity}")


    return characters

def build_network_from_file(file_path, entity_label=DEFAULT_ENTITY_LABEL):
    """ Builds a co-occurrence network from a single segmented NER file. """
    co_occurrence_counts = defaultdict(int)
    all_characters = set()

    logging.info(f"Processing file: {os.path.basename(file_path)}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            segmented_data = json.load(f)
    except json.JSONDecodeError:
        logging.error(f"Failed to decode JSON from {file_path}. Skipping.")
        return nx.Graph(), False
    except Exception as e:
        logging.error(f"Failed to read file {file_path}: {e}. Skipping.")
        return nx.Graph(), False

    if not isinstance(segmented_data, dict):
        logging.warning(f"Expected dict in {file_path}, got {type(segmented_data)}. Skipping.")
        return nx.Graph(), False


    for segment_id, entity_list in segmented_data.items():
        chars_in_segment = extract_characters(entity_list, entity_label)
        if not chars_in_segment:
            continue

        all_characters.update(chars_in_segment)

        # Increment co-occurrence count for pairs within this segment
        if len(chars_in_segment) >= 2:
            # Sort to ensure consistent pair ordering (e.g., (A, B) not (B, A))
            sorted_chars = sorted(list(chars_in_segment))
            for char1, char2 in combinations(sorted_chars, 2):
                co_occurrence_counts[(char1, char2)] += 1


    logging.info(f"Building NetworkX graph... Found {len(all_characters)} unique characters.")
    G = nx.Graph()
    G.add_nodes_from(all_characters)

    logging.info(f"Adding {len(co_occurrence_counts)} edges based on co-occurrence counts.")
    for (char1, char2), weight in co_occurrence_counts.items():
        G.add_edge(char1, char2, weight=weight)

    logging.info(f"Network built successfully: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G, True

def save_graph(graph, output_path):
    """Saves the graph to a file. Detects format from extension."""
    _, ext = os.path.splitext(output_path)
    ext = ext.lower()

    try:
        if ext == '.gexf':
            nx.write_gexf(graph, output_path)
        elif ext == '.graphml':
            nx.write_graphml(graph, output_path)
        elif ext == '.json':
             from networkx.readwrite import json_graph
             data = json_graph.node_link_data(graph)
             with open(output_path, 'w', encoding='utf-8') as f:
                 json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            logging.error(f"Unsupported output format: {ext}. Use .gexf, .graphml, or .json")
            return False
        logging.info(f"Graph saved successfully to {output_path}")
        return True
    except Exception as e:
        logging.error(f"Failed to save graph to {output_path}: {e}")
        return False

project_root = get_project_root()
default_input_dir = os.path.join(project_root, "results", "extracted_entities")
default_output_dir = os.path.join(project_root, "results", "networks")

parser = argparse.ArgumentParser(description="Build character co-occurrence networks from segmented NER results.")
parser.add_argument(
    "--input_dir",
    type=str,
    default=default_input_dir,
    help=f"Directory containing the '*_ner_segmented.json' files (default: {default_input_dir})"
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=default_output_dir,
    help=f"Directory to save the output graph files (default: {default_output_dir})"
)
parser.add_argument(
    "--output_extension",
    type=str,
    default=DEFAULT_OUTPUT_EXTENSION,
    help=f"File extension for output files (gexf, graphml, json) (default: {DEFAULT_OUTPUT_EXTENSION})"
)
parser.add_argument(
    "--entity_label",
    type=str,
    default=DEFAULT_ENTITY_LABEL,
    help=f"The entity label used for characters in the NER results (default: '{DEFAULT_ENTITY_LABEL}')"
)
parser.add_argument(
    "--combined",
    action="store_true",
    help="Also generate a combined network from all files"
)

args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# Make sure the extension doesn't include the dot
output_extension = args.output_extension.lower()
if output_extension.startswith('.'):
    output_extension = output_extension[1:]

try:
     ner_files = [f for f in os.listdir(args.input_dir) if f.endswith("_ner_segmented.json")]
     if not ner_files:
          logging.warning(f"No '*_ner_segmented.json' files found in {args.input_dir}. Cannot build network.")
          # Exit gracefully if no files found
          exit()
except FileNotFoundError:
     logging.error(f"Input directory not found: {args.input_dir}")
     # Exit gracefully if input directory not found
     exit()

processed_count = 0
combined_graph = nx.Graph() if args.combined else None

for filename in ner_files:
    file_path = os.path.join(args.input_dir, filename)

    text_name = filename.replace("_ner_segmented.json", "")
    output_filename = f"{text_name}_network.{output_extension}"
    output_path = os.path.join(args.output_dir, output_filename)

    graph, success = build_network_from_file(file_path, args.entity_label)

    if success and graph.number_of_nodes() > 0:
        save_graph(graph, output_path)
        processed_count += 1

        if args.combined and combined_graph is not None:
            # Explicitly check if combined_graph is not None before using compose
            # Use nx.compose to merge graphs. Weights aren't automatically summed.
            # We need to manually handle edge weights for composed graphs.
            nodes_to_add = [n for n in graph.nodes() if n not in combined_graph]
            combined_graph.add_nodes_from(nodes_to_add)

            for u, v, d in graph.edges(data=True):
                edge_weight = d.get('weight', 1) # Default weight to 1 if missing
                if combined_graph.has_edge(u, v):
                    # If edge exists, add to the existing weight
                    combined_graph[u][v]['weight'] = combined_graph[u][v].get('weight', 0) + edge_weight
                else:
                    # If edge doesn't exist, add it with its weight
                    combined_graph.add_edge(u, v, weight=edge_weight)

# Save combined graph if requested
if args.combined and combined_graph is not None and combined_graph.number_of_nodes() > 0:
    combined_path = os.path.join(args.output_dir, f"combined_network.{output_extension}")
    if save_graph(combined_graph, combined_path):
        logging.info(f"Combined network saved to {combined_path}")

logging.info(f"--- Network Building Script Finished ---")
logging.info(f"Created {processed_count} individual character networks in {args.output_dir}") 