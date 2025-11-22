"""fedrag: A Flower Federated RAG app."""

import hashlib
import time
from collections import defaultdict
from itertools import cycle
from pathlib import Path
from time import sleep
import sys

import numpy as np
from flwr.common import ConfigRecord, Context, Message, MessageType, RecordDict
from flwr.server import Grid, ServerApp
from sklearn.metrics import accuracy_score

from fedrag.llm_querier import LLMQuerier
from fedrag.mirage_qa import MirageQA

# MirageQA dataset path will be injected via config at runtime
# from fedrag_adapter.py - this ensures the path works both in dev
# and when the code is copied to RDS/SyftBox network directories


def node_online_loop(grid: Grid) -> list[int]:
    node_ids = []
    while not node_ids:
        # Get IDs of nodes available
        node_ids = grid.get_node_ids()
        # Wait if no node is available
        sleep(1)
    return node_ids


def get_hash(doc):
    # Create and return an SHA-256 hash for the given document
    return hashlib.sha256(doc.encode())


def merge_documents(documents, scores, knn, k_rrf=60, reverse_sort=False) -> list[str]:
    RRF_dict = defaultdict(dict)
    sorted_scores = np.array(scores).argsort()
    if reverse_sort:  # from larger to smaller scores
        sorted_scores = sorted_scores[::-1]
    sorted_documents = [documents[i] for i in sorted_scores]

    if k_rrf == 0:
        # If k_rff is not set then simply return the
        # sorted documents based on their retrieval score
        return sorted_documents[:knn]
    else:
        for doc_idx, doc in enumerate(sorted_documents):
            # Given that some returned results/documents could be extremely
            # large we cannot use the original document as a dictionary key.
            # Therefore, we first hash the returned string/document to a
            # representative hash code, and we use that code as a key for
            # the final RRF dictionary. We follow this approach, because a
            # document could  have been retrieved twice by multiple clients
            # but with different scores and depending on these scores we need
            # to maintain its ranking
            doc_hash = get_hash(doc)
            RRF_dict[doc_hash]["rank"] = 1 / (k_rrf + doc_idx + 1)
            RRF_dict[doc_hash]["doc"] = doc

        RRF_docs = sorted(RRF_dict.values(), key=lambda x: x["rank"], reverse=True)
        docs = [rrf_res["doc"] for rrf_res in RRF_docs][
            :knn
        ]  # select the final top-k / k-nn
        return docs


def submit_question(
    grid: Grid,
    question: str,
    question_id: str,
    knn: int,
    node_ids: list,
    corpus_names_iter: iter,
):
    messages = []
    # Send the same Message to each connected node (which run `ClientApp` instances)
    for node_idx, node_id in enumerate(node_ids):
        # The payload of a Message is of type RecordDict
        # https://flower.ai/docs/framework/ref-api/flwr.common.RecordDict.html
        # which can carry different types of records. We'll use a ConfigRecord object
        # We need to create a new ConfigRecord() object for every node, otherwise
        # if we just override a single key, e.g., corpus_name, the grid will send
        # the same object to all nodes.
        config_record = ConfigRecord()
        config_record["question"] = question
        config_record["question_id"] = question_id
        config_record["knn"] = knn
        # Round-Robin assignment of corpus to individual clients
        # by infinitely looping over the corpus names.
        config_record["corpus_name"] = next(corpus_names_iter)

        task_record = RecordDict({"config": config_record})
        message = Message(
            content=task_record,
            message_type=MessageType.QUERY,  # target `query` method in ClientApp
            dst_node_id=node_id,
            group_id=str(question_id),
        )
        messages.append(message)

    # Calculate size of outgoing messages (query)
    query_size_bytes = sum(sys.getsizeof(str(msg)) for msg in messages)

    # Send messages and wait for all results
    replies = grid.send_and_receive(messages)
    print(f"✓ Received {len(replies)}/{len(messages)} results")

    documents, scores = [], []
    for reply in replies:
        if reply.has_content():
            documents.extend(reply.content["docs_n_scores"]["documents"])
            scores.extend(reply.content["docs_n_scores"]["scores"])

    # Calculate size of incoming messages (retrieved documents)
    response_size_bytes = sum(
        sys.getsizeof(str(doc)) for doc in documents
    ) + sys.getsizeof(str(scores))

    # Total communication size in MB
    total_comm_size_mb = (query_size_bytes + response_size_bytes) / (1024 * 1024)

    return documents, scores, total_comm_size_mb


app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    node_ids = node_online_loop(grid)

    # k-reciprocal-rank-fusion is used by the server to merge
    # the results returned by the clients
    k_rrf = int(context.run_config["k-rrf"])
    # k-nearest-neighbors for document retrieval at each client
    knn = int(context.run_config["k-nn"])
    corpus_names = context.run_config["clients-corpus-names"].split("|")
    corpus_names = [c.lower() for c in corpus_names]  # make them lower case

    # Create corpus iterator
    corpus_names_iter = cycle(corpus_names)
    qa_datasets = context.run_config["server-qa-datasets"].split("|")
    qa_datasets = [qa_d.lower() for qa_d in qa_datasets]  # make them lower case
    qa_num = context.run_config.get("server-qa-num", None)
    model_name = context.run_config["server-llm-hfpath"]
    use_gpu = context.run_config.get("server-llm-use-gpu", False)
    use_gpu = True if use_gpu.lower() == "true" else False
    max_new_tokens = int(context.run_config.get("server-llm-max-new-tokens", 50))

    # Get MirageQA dataset path from config (injected by fedrag_adapter)
    mirage_file_path = context.run_config["server-mirage-qa-path"]
    mirage_file = Path(mirage_file_path)

    # Auto-download MirageQA dataset if not found
    if not mirage_file.exists():
        print(f"⬇️  MirageQA dataset not found at {mirage_file}")
        print(f"📥 Downloading from {MirageQA.RAW_JSON_FILE}...")
        mirage_file.parent.mkdir(parents=True, exist_ok=True)
        MirageQA.download(mirage_file)
        print(f"✅ Downloaded MirageQA dataset to {mirage_file}")

    datasets = {key: MirageQA(key, mirage_file) for key in qa_datasets}

    llm_querier = LLMQuerier(model_name, use_gpu)
    expected_answers, predicted_answers, question_times, unanswered_questions = (
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
        defaultdict(int),
    )
    communication_sizes = defaultdict(list)  # Track communication size (MB) per dataset

    # Accuracy breakdown tracking
    option_correct = defaultdict(lambda: defaultdict(int))  # [dataset][option] = count
    option_total = defaultdict(lambda: defaultdict(int))  # [dataset][option] = count
    option_predictions = defaultdict(
        lambda: defaultdict(int)
    )  # [dataset][option] = count
    confusion_matrix = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )  # [dataset][expected][predicted] = count
    for dataset_name in qa_datasets:
        q_idx = 0
        print(f"\n{'='*60}")
        print(f"📊 Evaluating Dataset: [{dataset_name}]")
        print(f"{'='*60}")
        for q in datasets[dataset_name]:
            q_idx += 1
            q_id = f"{dataset_name}_{q_idx}"
            # exit question loop if number of questions has been exceeded
            if qa_num and q_idx > qa_num:
                break
            question = q["question"]
            q_st = time.time()
            docs, scores, comm_size_mb = submit_question(
                grid, question, q_id, knn, node_ids, corpus_names_iter
            )
            merged_docs = merge_documents(docs, scores, knn, k_rrf)
            options = q["options"]
            answer = q["answer"]

            prompt, predicted_answer = llm_querier.answer(
                question, merged_docs, options, dataset_name, max_new_tokens
            )

            # If the model did not predict any value,
            # then discard the question
            if predicted_answer is not None:
                expected_answers[dataset_name].append(answer)
                predicted_answers[dataset_name].append(predicted_answer)
                q_et = time.time()
                q_time = q_et - q_st  # elapsed time in seconds
                question_times[dataset_name].append(q_time)
                communication_sizes[dataset_name].append(comm_size_mb)

                # Track accuracy breakdown metrics
                option_total[dataset_name][answer] += 1
                option_predictions[dataset_name][predicted_answer] += 1
                if predicted_answer == answer:
                    option_correct[dataset_name][answer] += 1
                confusion_matrix[dataset_name][answer][predicted_answer] += 1
            else:
                unanswered_questions[dataset_name] += 1

    print("\n" + "=" * 80)
    print("📈 FEDERATED RAG EVALUATION RESULTS")
    print("=" * 80)
    print("\nMetrics Explained:")
    print("  (1) Total Questions - Number of Federated RAG queries executed")
    print("  (2) Answered Questions - Queries answered by LLM with retrieved documents")
    print("  (3) Accuracy - Expected answer vs. LLM predicted answer")
    print(
        "  (4) Mean Querying Time - Average wall-clock time per query (submission → final prediction)"
    )
    print(
        "  (5) Communication Cost - Average data exchanged per query (query + response)"
    )
    print("=" * 80 + "\n")
    for dataset_name in qa_datasets:
        exp_ans = expected_answers[dataset_name]
        pred_ans = predicted_answers[dataset_name]
        not_answered = unanswered_questions[dataset_name]
        total_questions = len(exp_ans) + not_answered
        accuracy = 0.0
        if exp_ans and pred_ans:  # make sure that both collections have values inside
            accuracy = accuracy_score(exp_ans, pred_ans)
        elapsed_time = np.mean(question_times[dataset_name])
        mean_comm_size = np.mean(communication_sizes[dataset_name])
        total_comm_size = np.sum(communication_sizes[dataset_name])

        # Print dataset results with nice formatting
        print(f"\n{'─'*60}")
        print(f"🔍 QA Dataset: {dataset_name}")
        print(f"{'─'*60}")
        print(f"  Total Questions:     {total_questions}")
        print(f"  Answered Questions:  {len(pred_ans)}")
        print(f"  Accuracy:            {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Mean Querying Time:  {elapsed_time:.2f}s")
        print(f"  Mean Comm. Cost:     {mean_comm_size:.4f} MB/query")
        print(f"  Total Comm. Cost:    {total_comm_size:.2f} MB")
        print(f"{'─'*60}")

        # Print accuracy breakdown
        if len(pred_ans) > 0:
            print(f"\n📊 Accuracy Breakdown: {dataset_name}")

            # Per-option accuracy
            print("\n  Per-Option Accuracy:")
            all_options = sorted(
                set(
                    list(option_total[dataset_name].keys())
                    + list(option_predictions[dataset_name].keys())
                )
            )
            for option in all_options:
                correct = option_correct[dataset_name].get(option, 0)
                total = option_total[dataset_name].get(option, 0)
                acc = correct / total if total > 0 else 0.0
                print(f"    Option {option}: {acc:.4f} ({correct}/{total})")

            # Option distribution (predictions)
            print("\n  Option Distribution (Predicted):")
            total_predictions = sum(option_predictions[dataset_name].values())
            if total_predictions > 0:
                option_percentages = []
                for option in all_options:
                    count = option_predictions[dataset_name].get(option, 0)
                    percentage = (count / total_predictions) * 100
                    option_percentages.append(f"{option}: {percentage:.1f}%")
                print("    " + ", ".join(option_percentages))

            # Confusion Matrix
            print("\n🧮 Confusion Matrix:")
            print("         " + "    ".join(all_options))
            for expected_opt in all_options:
                row_values = []
                for predicted_opt in all_options:
                    count = confusion_matrix[dataset_name][expected_opt].get(
                        predicted_opt, 0
                    )
                    row_values.append(f"{count:4d}")
                print(f"    {expected_opt}    " + "    ".join(row_values))
            print()  # Empty line for separation
