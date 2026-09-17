# Sparse Attention Routing for Long-Document Retrieval-Augmented Generation

**Authors:** Elena Marchetti (Politecnico di Torino), Kwame Mensah (University of Ghana), Sofia Lindqvist (KTH Royal Institute of Technology)

**Venue:** EMNLP 2024

**Year:** 2024

**DOI:** 10.0000/synthetic.2024.001

## Abstract

Retrieval-Augmented Generation systems struggle when the evidence needed to answer a question is scattered across very long documents. We introduce Sparse Attention Routing, a method that partitions a long document into evidence regions and learns a routing distribution over those regions before the decoder attends to them. Sparse Attention Routing combines Dual-Level Retrieval, which pairs low-level entity lookup with high-level thematic aggregation, with a Transformer decoder whose attention budget is allocated proportionally to the routing weights. On NarrativeQA and LongBench the approach improves Exact Match by 4.1 points and ROUGE-L by 3.6 points over a strong Retrieval-Augmented Generation baseline while attending to 38 percent fewer tokens.

## 1 Introduction

Long-Document Understanding remains one of the central open problems for Retrieval-Augmented Generation. Standard pipelines retrieve a fixed number of passages, concatenate them, and hand the result to a generator. When a document exceeds the context window, the retriever has to choose between coverage and precision, and the generator has no mechanism to recover evidence that the retriever dropped.

Graph-based approaches address part of this gap. LightRAG organises extracted entities and relations into a Knowledge Graph and retrieves along two levels: specific entities for local questions and abstract themes for global questions. GraphRAG builds community summaries over the same kind of graph and answers query-focused summarisation tasks by reading those summaries. Both systems show that structure helps, but both still hand the generator a flat block of text.

We argue that the missing ingredient is routing. Sparse Attention Routing treats each retrieved region as a candidate and learns how much decoder attention each region deserves. The routing distribution is sparse, so most regions receive no attention at all, which keeps the compute budget bounded even when the document is very long.

## 2 Method: Sparse Attention Routing

Sparse Attention Routing operates in three stages.

**Stage one: region construction.** The document is segmented into regions of roughly 400 tokens. Each region is indexed twice, once by dense embedding for low-level lookup and once by the set of Knowledge Graph entities it mentions for high-level aggregation. This is the same Dual-Level Retrieval principle popularised by LightRAG, but applied inside a single long document rather than across a corpus.

**Stage two: routing.** A lightweight router network scores every region against the question and produces a distribution using a sparsemax projection. Regions with zero routing mass are discarded. In practice between four and nine regions survive for a typical NarrativeQA question.

**Stage three: budgeted decoding.** A Transformer decoder attends to the surviving regions. The number of attention heads assigned to each region is proportional to its routing weight, so the decoder spends most of its capacity on the evidence the router considers most relevant. The decoder is initialised from a standard Transformer checkpoint and fine-tuned jointly with the router.

The router and decoder are trained with a combined objective: a generation loss on the gold answer and an auxiliary routing loss that rewards placing mass on regions containing gold evidence spans.

## 3 Experiments

**Datasets.** We evaluate on NarrativeQA, where questions are asked about entire books and film scripts, and on LongBench, a multi-task benchmark for long-context understanding. Both datasets contain documents that exceed 16,000 tokens.

**Metrics.** We report Exact Match and ROUGE-L. Exact Match measures whether the generated answer matches a gold answer after normalisation; ROUGE-L measures longest-common-subsequence overlap and rewards partially correct answers.

**Baselines.** We compare against a Retrieval-Augmented Generation baseline that retrieves the top eight passages, a LightRAG configuration using its hybrid mode, a GraphRAG configuration using community summaries, and a plain Transformer that reads a truncated document.

**Results.** Sparse Attention Routing reaches 31.7 Exact Match on NarrativeQA compared with 27.6 for the Retrieval-Augmented Generation baseline and 29.4 for LightRAG. On LongBench the method improves ROUGE-L from 41.2 to 44.8. The routing distribution is sparse: on average 6.2 regions receive non-zero mass out of 41.

**Ablations.** Removing the auxiliary routing loss costs 1.9 Exact Match. Replacing sparsemax with softmax removes the sparsity and increases attended tokens by 61 percent with no accuracy gain. Removing the Knowledge Graph index and relying on dense retrieval alone costs 2.3 ROUGE-L on LongBench, confirming that Dual-Level Retrieval matters even inside a single document.

## 4 Related Work

The Transformer architecture introduced in Attention Is All You Need is the foundation of every generator considered here. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks established the retrieve-then-generate paradigm we build on. LightRAG: Simple and Fast Retrieval-Augmented Generation introduced the Dual-Level Retrieval scheme that our region index adopts, and From Local to Global: A Graph RAG Approach to Query-Focused Summarization demonstrated the value of Knowledge Graph community structure for global questions. Sparse Attention Routing differs from all of these by learning where the decoder should look rather than only what the retriever should return.

## 5 Conclusion

Sparse Attention Routing shows that Long-Document Understanding benefits from an explicit routing step between retrieval and generation. By combining Dual-Level Retrieval with a budgeted Transformer decoder, the method improves Exact Match and ROUGE-L on NarrativeQA and LongBench while attending to substantially fewer tokens. Future work will extend the router to multi-document settings and study its interaction with Knowledge Graph community summaries.

## Acknowledgements

Elena Marchetti was supported by Politecnico di Torino. Kwame Mensah was supported by the University of Ghana. Sofia Lindqvist was supported by KTH Royal Institute of Technology.

## References

[1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., and Polosukhin, I. (2017). Attention Is All You Need. Advances in Neural Information Processing Systems 30.

[2] Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Kuttler, H., Lewis, M., Yih, W., Rocktaschel, T., Riedel, S., and Kiela, D. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. Advances in Neural Information Processing Systems 33.

[3] Guo, Z., Xia, L., Yu, Y., Ao, T., and Huang, C. (2024). LightRAG: Simple and Fast Retrieval-Augmented Generation. arXiv preprint arXiv:2410.05779.

[4] Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., Truitt, S., and Larson, J. (2024). From Local to Global: A Graph RAG Approach to Query-Focused Summarization. arXiv preprint arXiv:2404.16130.
