from pylate import indexes, models, retrieve

# model = models.ColBERT(
#     model_name_or_path="lightonai/GTE-ModernColBERT-v1",
# )

model = models.ColBERT(
    model_name_or_path="colbert-ir/colbertv2.0",
)


index = indexes.PLAID(
    index_folder="pylate-index",
    index_name="index",
    override=True,
)
retriever = retrieve.ColBERT(index=index)
documents_ids = ["1", "2", "3"]

documents = [
    "ColBERT’s late-interaction keeps token-level embeddings to deliver cross-encoder-quality ranking at near-bi-encoder speed, enabling fine-grained relevance, robustness across domains, and hardware-friendly scalable search.",
    "PLAID compresses ColBERT token vectors via product quantization to shrink storage by 10×, uses two-stage centroid scoring for sub-200 ms latency, and plugs directly into existing ColBERT pipelines.",
    "PyLate is a library built on top of Sentence Transformers, designed to simplify and optimize fine-tuning, inference, and retrieval with state-of-the-art ColBERT models. It enables easy fine-tuning on both single and multiple GPUs, providing flexibility for various hardware setups. PyLate also streamlines document retrieval and allows you to load a wide range of models, enabling you to construct ColBERT models from most pre-trained language models.",
]
# Encode the documents
documents_embeddings = model.encode(
    documents,
    batch_size=32,
    # pool_factor=2, # to reduce index footprint
    is_query=False,
    show_progress_bar=True,
)
# Add the documents ids and embeddings to the PLAID index
index.add_documents(
    documents_ids=documents_ids,
    documents_embeddings=documents_embeddings,
)
queries_embeddings = model.encode(
    ["What is ColBERT?", "What is PyLate?", "What is late-interaction?"],
    batch_size=32,
    is_query=True,
    show_progress_bar=True,
)
scores = retriever.retrieve(
    queries_embeddings=queries_embeddings,
    k=5,
)

from rich import print
print(scores)
