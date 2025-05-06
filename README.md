# Armenian NLP Project: NER and Network Analysis

This project focuses on processing Armenian text data, performing Named Entity Recognition (NER), and building/visualizing networks based on the extracted entities.

## Project Structure

```
Armenian_ML/
├── data/
│   ├── processed/      # Processed text data
│   └── raw/            # Raw text data downloaded
├── external_repos/     # External repositories (e.g., fairseq, pioner)
├── models/             # (Used during dev; excluded from repo - see Hugging Face link below)
├── results/
│   ├── extracted_entities/ # Output of NER inference
│   ├── network_visualizations/ # Network graphs
│   └── networks/       # Saved network data (e.g., GraphML, edge lists)
├── scripts/
│   ├── 01_data_acquisition/
│   │   └── download_data.py  # Script to download necessary datasets
│   ├── 02_preprocessing/
│   │   └── preprocess_texts.py # Script for text cleaning and preparation
│   ├── 03_ner/
│   │   └── run_ner_inference_segmented.py # Script to run NER inference on processed data
│   └── 04_network_building/
│       ├── build_network.py    # Script to construct networks from NER results
│       └── visualize_network.py # Script to visualize the constructed networks
├── README.md           # This file
└── requirements.txt    # Project dependencies
```

## Pipeline Overview

The project follows a sequential pipeline implemented through the scripts in the `scripts/` directory:

1.  **Data Acquisition (`01_data_acquisition/download_data.py`):** 
    Downloads the initial raw data required for the project.
2.  **Preprocessing (`02_preprocessing/preprocess_texts.py`):** 
    Cleans and prepares the raw text data, likely involving steps like tokenization, normalization, and segmentation. The output is stored in `data/processed/`.
3.  **Named Entity Recognition (`03_ner/`):**
    *   Uses a model fine-tuned RoBERTa to identify named entities. The best performing model (`daviddallakyan2005/armenian-ner`) is hosted on [Hugging Face Hub](https://huggingface.co/daviddallakyan2005/armenian-ner). (See `scripts/03_ner/ner_roberta.py` for potential details on original training/evaluation).
    *   _During development, 36 models with varying hyperparameters were evaluated (details in `scripts/03_ner/ner_roberta.py`), and the best model was selected based on its F1 score._
    *   `run_ner_inference_segmented.py` applies the NER model (loaded from Hugging Face) to the processed data, saving the extracted entities, likely into `results/extracted_entities/`.
4.  **Network Building (`04_network_building/`):**
    *   `build_network.py` uses the extracted entities to construct relationship networks (e.g., co-occurrence networks). The resulting network data might be saved in `results/networks/`.
    *   `visualize_network.py` generates visualizations of these networks, saving them in `results/network_visualizations/`.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/daviddallakyan2005/armenian-ner-network.git
    ```
2.  **Create and activate a virtual environment:**
    *   **macOS:**
        ```bash
        python3 -m venv env
        source env/bin/activate
        ```
    *   **Windows:**
        ```bash
        python -m venv venv
        .\\venv\\Scripts\\activate
        ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *   **PyTorch with CUDA:** If you require GPU acceleration with CUDA 12.1, ensure you install the correct PyTorch version. Uninstall any existing PyTorch versions first (`pip uninstall torch -y`) and then run:
        ```bash
        pip install torch --index-url https://download.pytorch.org/whl/cu121
        ```

## Usage

Execute the scripts sequentially, starting from data acquisition:

```bash
python scripts/01_data_acquisition/download_data.py
python scripts/02_preprocessing/preprocess_texts.py
python scripts/03_ner/run_ner_inference_segmented.py --model_name_or_path daviddallakyan2005/armenian-ner # Example param needed
python scripts/04_network_building/build_network.py # Params needed
python scripts/04_network_building/visualize_network.py
```

*Note: You may need to adjust script arguments (e.g., file paths) within the scripts or pass them via command-line arguments if supported. The NER script now points to the Hugging Face model by default.*

## Results

*   **Processed Data:** Located in `data/processed/`.
*   **NER Output:** Located in `results/extracted_entities/`.
*   **Network Data:** Located in `results/networks/`.
*   **Network Visualizations:** Located in `results/network_visualizations/`.

## External Repositories

This project utilizes code or models from the following external repositories, located in the `external_repos/` directory:
*   [`fairseq`](https://github.com/facebookresearch/fairseq.git)
*   [`pioner`](https://github.com/ispras-texterra/pioner.git)
*   [`ArmTokenizer`](https://github.com/DavidDavidsonDK/ArmTokenizer.git)

Refer to their respective documentation for more details if needed.
