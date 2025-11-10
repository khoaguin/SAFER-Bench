"""fedrag: A Flower Federated RAG app."""

import json
import warnings
from collections import OrderedDict
from pathlib import Path

import faiss
import numpy as np
import yaml
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from typing_extensions import Tuple

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

DIR_PATH = Path(__file__).resolve().parent
FAISS_DEFAULT_CONFIG = DIR_PATH / "retriever.yaml"


class Retriever:
    def __init__(self, config_file=None):
        if not config_file:
            self.config = yaml.safe_load(open(FAISS_DEFAULT_CONFIG, "r"))
        else:
            self.config = yaml.safe_load(open(config_file, "r"))
        # load the embedding model and define the embeddings dimensions
        # for the device placement of the SentenceTransformers model we resort
        # to use the device name returned by `sentence_transformers.util.get_device_name()`
        # which will be called by the SentenceTransformer constructor when creating the model
        self.emb_model = SentenceTransformer(self.config["embedding_model"])
        self.emb_dim = self.config["embedding_dimension"]

    def build_faiss_index(self, dataset_name, batch_size=32, num_chunks=None):
        index_path, doc_ids_path, chunk_dir = _get_dataset_dirs(dataset_name)

        try:
            # erase previous files whenever
            # index builder is called
            index_path.unlink()
            doc_ids_path.unlink()
        except (OSError, FileNotFoundError):
            pass

        all_embeddings, all_doc_ids = [], []
        all_files = list(chunk_dir.glob("*.jsonl"))  # get all jsonl files
        # if chunks is given just load the specified
        # number of chunks; useful for dev and debug purposes
        if num_chunks:
            all_files = all_files[:num_chunks]

        # Log start of index building
        print(f"[DO] 🔨 Building FAISS index for dataset: {dataset_name}")
        print(f"[DO] 📁 Found {len(all_files)} chunk files to process")

        # Loop through all the .jsonl files, load the id and the content of
        # each document and for each document generate its embeddings
        for filename in tqdm(all_files):
            batch_content, batch_ids = [], []
            with open(filename, "r", encoding="utf-8") as infile:
                for line in infile:
                    doc = json.loads(line)
                    doc_id = doc.get("id", "")
                    content = doc.get("content", "")
                    batch_ids.append(doc_id)
                    batch_content.append(content)

                    if len(batch_ids) > batch_size:
                        # Generate embeddings for the batch
                        batch_embeddings = self.emb_model.encode(
                            batch_content, convert_to_numpy=True
                        )
                        all_embeddings.extend(batch_embeddings)
                        all_doc_ids.extend(batch_ids)
                        batch_content, batch_ids = [], []

                # Process last batch
                if batch_content:
                    batch_embeddings = self.emb_model.encode(
                        batch_content, convert_to_numpy=True
                    )
                    all_embeddings.extend(batch_embeddings)
                    all_doc_ids.extend(batch_ids)

        # Filter out embeddings if they do not have the expected dimensions
        filtered_embeddings = [
            embedding
            for embedding in all_embeddings
            if embedding is not None and embedding.shape == (self.emb_dim,)
        ]
        # FAISS needs float32, hence the casting
        embeddings = np.array(filtered_embeddings).astype("float32")
        d = embeddings.shape[1]  # Dimensionality of the embeddings

        # Quantizer for IVF
        quantizer = faiss.IndexFlatL2(d)

        # Number of clusters
        nlist = int(np.sqrt(len(embeddings)))

        # METRIC_L2 measures dissimilarity, hence the lower the score the better!
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

        # Train the index
        index.train(embeddings)

        # Add the embeddings to the index
        index.add(embeddings)

        # Save the index
        faiss.write_index(index, str(index_path))

        # Save document IDs
        np.save(str(doc_ids_path), np.array(all_doc_ids))

        print(f"[DO] ✅ FAISS index built successfully for {dataset_name}")

        return

    def query_faiss_index(self, dataset_name, query, knn=8):
        # Check if index exists, build if needed
        if not self.index_exists(dataset_name):
            print(f"FAISS index not found for {dataset_name}. Building index...")
            self.build_faiss_index(dataset_name)
            print(f"FAISS index built successfully for {dataset_name}")

        index_path, doc_ids_path, chunk_dir = _get_dataset_dirs(dataset_name)

        # 1. Load the FAISS index and document IDs
        index = faiss.read_index(str(index_path))
        doc_ids = np.load(str(doc_ids_path))

        # 2. Generate query embedding
        query_embedding = self.emb_model.encode(query)

        # 3. Search the index
        # CAUTION: since our FAISS index is built with
        # IndexIVFFlat and metric faiss.METRIC_L2, the
        # lower the score the better, since L2 Distance
        # measures dissimilarity.
        doc_scores, doc_idx = index.search(np.array([query_embedding]), knn)

        # 4. Retrieve the relevant document IDs
        doc_scores = doc_scores[0]  # flatten scores
        retrieved_doc_ids = doc_ids[doc_idx][0]  # flatten ids

        # 5. Prepare and return the results
        final_res = OrderedDict()
        for i, (doc_id, doc_score) in enumerate(zip(retrieved_doc_ids, doc_scores)):
            doc_pref_suf = doc_id.split("_")
            doc_name, snippet_idx = "_".join(doc_pref_suf[:-1]), int(doc_pref_suf[-1])

            # Try exact match first (for single dataset scenarios)
            full_file = chunk_dir / (doc_name + ".jsonl")

            # If not found, search for prefixed version (for centralized/hybrid partitions)
            # e.g., statpearls_article-31824.jsonl
            if not full_file.exists():
                matching_files = list(chunk_dir.glob(f"*_{doc_name}.jsonl"))
                if matching_files:
                    full_file = matching_files[0]
                else:
                    raise FileNotFoundError(
                        f"Cannot find chunk file for {doc_name} in {chunk_dir}"
                    )

            loaded_snippet = json.loads(
                open(full_file).read().strip().split("\n")[snippet_idx]
            )
            rank = i + 1
            final_res[doc_id] = {
                "rank": int(rank),
                "score": float(doc_score),
                "title": str(loaded_snippet["title"]),
                "content": str(loaded_snippet["content"]),
            }

        return final_res

    @classmethod
    def index_exists(cls, dataset_name):
        index_path, doc_ids_path, _ = _get_dataset_dirs(dataset_name)
        return index_path.exists() and doc_ids_path.exists()


def _get_dataset_dirs(dataset_name: str) -> Tuple[Path, Path, Path]:
    """
    Return index, doc ids and chunk dirs.

    Note: Index and doc_ids files may not exist yet for partitioned datasets.
    They will be built on-demand by the Retriever when first queried.
    """

    from syft_flwr.utils import get_syftbox_dataset_path, run_syft_flwr

    if not run_syft_flwr():
        data_dir = DIR_PATH.parent.parent / "data" / "corpus" / dataset_name
    else:
        data_dir = get_syftbox_dataset_path()

    index_path = data_dir / "faiss.index"
    doc_ids_path = data_dir / "all_doc_ids.npy"
    chunk_dir = data_dir / "chunk"

    # Only assert chunk directory exists (required for building index)
    assert chunk_dir.exists(), f"Chunk directory {chunk_dir} does not exist"

    return index_path, doc_ids_path, chunk_dir
