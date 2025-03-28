# Clinical_ModernBERT
---

We introduce Clinical ModernBERT, a novel transformer-based encoder pretrained on large-scale biomedical literature and clinical notes, leveraging both PubMed abstracts and MIMIC-IV data. Building upon ModernBERT, which represents the current state-of-the-art in encoder efficiency through innovations such as RoPE positional encoding, flash attention, and extended context length, our model adapts these advancements specifically to biomedical and clinical domains. Although recent progress in generative language modeling has predominantly focused on decoder-based architectures, we argue that encoder-based transformers remain highly relevant and effective—particularly in biomedical and clinical applications. Models such as Clinical ModernBERT inherently excel at producing semantically rich representations, essential for tasks including retrieval-augmented generation (RAG), commonly employed in evidence-based clinical protocols; fine-grained text classification of long-range patient narratives; and domain-specific information extraction tasks benefiting from enhanced representational capacity.

---

# environment

`conda env create -f environment.yml`

# run code

`CUDA_VISIBLE_DEVICES=1 python3 1percent.py`

`CUDA_VISIBLE_DEVICES=1 python3 main.py`

---

# data

- https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/
- https://www.physionet.org/content/mimic-iv-note/2.2/

---

# resources

- https://arxiv.org/pdf/2412.13663
- https://github.com/gatech-sysml/examples-mosaic-bert
- https://mosaicbert.github.io/

---

# tracking

- https://wandb.ai/home
