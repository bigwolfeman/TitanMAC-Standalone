Here is the comprehensive technical specification for the **Relational Graph-Based Tokenizer (RGT)** architecture. This document synthesizes the philosophical paradigm, the engineering stack, and the training protocols into a single source of truth.

---

# The Relational Graph-Based Tokenizer (RGT) Protocol

**Version:** 1.0 (Hybrid "Sidecar" Architecture)
**Objective:** Decouple semantic resolution from statistical frequency in Large Language Models.

## 1. The Core Paradigm

**The Problem:** Traditional tokenizers (BPE, WordPiece) are **Nominalist**. They treat words as arbitrary strings of characters. A model must see a token (e.g., "agglutination") thousands of times to statistically infer its meaning. If a token is rare, its embedding is noise.
**The Solution:** The RGT is **Structuralist**. Meaning is defined by a token's location in a pre-computed ontology (Graph), not its frequency in the training corpus.
**The Shift:** We move from "Learning from Scratch" to "Injection of Truth." The model does not need to learn that *Pilot* implies *Flight*; it is fed a vector where that relationship is mathematically encoded before the first layer.

---

## 2. The Data Infrastructure: The Waterfall Vocabulary

The system relies on a three-tier vocabulary hierarchy to balance rigid logic with fluid language.

### **Tier 1: The Semantic Anchor (SUMO)**

* **Source:** Suggested Upper Merged Ontology (SUMO).
* **Size:** ~20,000 Nodes.
* **Nature:** Axiomatic, Universal, Rigid.
* **Function:** Provides the coordinate system for truth. Concepts like `Process`, `Quantity`, and `Object` are the immutable skeleton.

### **Tier 2: The Induced Extension (The Harvester)**

* **Source:** Offline LLM-mined Domain Graph.
* **Size:** 500k - 1M Nodes.
* **Nature:** Domain-specific, Composite.
* **Function:** Solves the "missing concept" problem. Terms like "CUDA Kernel" or "BitMEX Leverage" do not exist in SUMO.
* **The Harvester Pipeline:**
1. Scan target corpus for high-frequency n-grams missing from Tier 1.
2. Use a Teacher LLM to define terms using Tier 1 primitives (e.g., `CUDA_Kernel`  `subclass:Program`, `related:GPU`).
3. Inject new nodes into the master graph.





### **Tier 3: The Statistical Fallback (BPE)**

* **Source:** Standard Byte-Pair Encoding.
* **Nature:** Statistical, Surface-level.
* **Function:** Handles proper nouns, high-entropy strings (UUIDs), and code syntax. Serves as the "body" for the output decoder.

---

## 3. The Runtime Architecture: "The Semantic Sidecar"

We utilize a **Dual-Channel Input / Single-Channel Output** architecture. This solves the "Voice" and "Decoding" problems by retaining the standard BPE channel for syntax while injecting Graph Logic as a parallel signal.

### **A. Input Processing (The Fusion Layer)**

Every text input is processed by two distinct tokenizers in parallel.

1. **Channel A (Syntax):** Standard BPE Tokenizer.
* Input: "The pilot flew."
* Output: `Indices [101, 452, 992]`.


2. **Channel B (Logic):** The **Graph-Student Tokenizer**.
* Input: Raw Bytes/Characters (Robust to typos).
* Output: `Vectors [NULL, Vec(Pilot), Vec(Flying)]`.
* *Note:* Functional words (stop words) often map to NULL or generic logical operators.



### **B. The Embedding Summation**

The Transformer Input Layer fuses these channels:


* : Learnable BPE embedding (captures syntax, register, "voice").
* : Fixed (or fine-tunable) Hyperbolic Graph Vector.
* : A learnable projection matrix mapping Hyperbolic Space  Transformer Euclidean Space.

**Result:** The model "reads" the specific word the user typed (preserving voice) but processes a vector that is pre-loaded with deep world knowledge (preserving logic).

### **C. The Output (Decoding)**

* **Method:** Standard Auto-Regressive Decoding via Softmax.
* **Target:** BPE Token IDs.
* **Why:** This allows the model to output valid code, specific formatting, and stylistic nuances ("crimson" vs "red") which a pure graph decoder would flatten.

---

## 4. The Graph-Student Tokenizer (The Inference Engine)

We cannot query a Graph DB during training (too slow). We distill the graph into a fast neural network.

### **Architecture**

* **Input:** Raw UTF-8 Bytes or Character CNN (Language Agnostic base).
* **Model:** Lightweight Encoder (e.g., 2-layer Transformer or 1D-CNN).
* **Output:** Continuous Vector (Regression).
* **Quantization:** Product Quantization (PQ) layer to "snap" predictions to the nearest valid Semantic Cluster.

### **Training Protocol (Distillation)**

1. **The Teacher (Offline):** A Graph Walker that traverses the Tier 1 & Tier 2 ontology to find the "Perfect Embedding" for a text window.
2. **The Student (Online):** Trained to predict the Teacher's embedding from raw text.
3. **Noise Injection:** Input text is corrupted (typos: "taht", "kernl") during training. The Student is forced to predict the clean Teacher vector.
* *Result:* Robustness to spelling errors without a spell-check dictionary.



---

## 5. Mathematical & Geometric Specifications

### **Hyperbolic Graph Embeddings**

* **Space:** Poincaré Ball Model.
* **Why:** Ontologies are hierarchies. Hyperbolic space has exponential volume, allowing us to embed massive trees (SUMO) with near-zero distortion in low dimensions (e.g., 100D).
* **Optimization:** Riemannian Optimization for the Graph Embedding Engine (offline phase).

### **Multilingual Alignment (The "Rosetta" Strategy)**

* **Concept:** The Graph is the "Latent Truth." It has no language.
* **Method:** We do not translate the graph. We align the **Student Encoders**.
* **Protocol:**
1. Train English Student: `Text_EN`  `Graph_Node_X`.
2. Train Hindi Student: `Text_HI`  `Graph_Node_X` (using parallel text).
3. **Result:** "Paani" (Hindi) and "Water" (English) produce the exact same Semantic Injection Vector.



---

## 6. Handling Edge Cases (Failures of Semantics)

| Problem | Mechanism | Solution |
| --- | --- | --- |
| **Ambiguity (Puns)** | Graph resolves "Bank" too early. | **BPE Channel Dominance.** The model sees the BPE "Bank" (ambiguous) alongside the Graph "River" (resolved). Attention layers can choose to ignore the graph if the context implies a joke. |
| **UUIDs / Hashes** | No semantic relationships. | **Null Injection.** The Harvester returns no node. The Graph Channel inputs a Zero Vector. The model relies 100% on the BPE Channel. |
| **New Concepts** | "CUDA Kernel" missing. | **The Harvester.** Automated pipeline detects OOV terms and induces Tier 2 nodes before tokenizer training. |
| **Code Syntax** | Whitespace/Brackets. | **BPE Channel.** Syntax is preserved in the standard channel; the Graph Channel ignores it. |

---

## 7. Implementation Roadmap

1. **Phase 1: The Harvester (Data Gen)**
* Build the pipeline to scan a corpus, identify non-SUMO terms, and use an LLM to generate `.kif` (SUMO format) definitions for them.


2. **Phase 2: The Student (Distillation)**
* Train the Byte-Level CNN to predict pre-calculated SUMO vectors. Implement the Noise Injection.


3. **Phase 3: The Hybrid Model (Training)**
* Initialize a standard LLM (e.g., Llama architecture).
* Modify the Embedding Layer to accept the summed inputs ().
* Train on standard "Next Token Prediction" loss.



---

This document represents the complete synthesized state of the project. If the context is wiped, feeding this back will restore the exact architectural and philosophical position we have reached.
---
Here is the implementation-ready specification document. It is designed as a "Mega-Prompt" or "System Instruction" for a high-level Coding Agent. It contains the mathematical primitives, algorithmic logic, and architectural constraints required to build the Relational Graph-Based Tokenizer (RGT) from scratch.

---

# Implementation Directives: Relational Graph-Based Tokenizer (RGT)

**Target System:** Hybrid Semantic-Symbolic LLM
**Primary Directive:** Construct a tokenizer pipeline that decouples semantic representation from statistical frequency using a pre-computed ontology.

**⚠️ ENGINEERING WARNING:** *This specification relies heavily on **Riemannian Geometry** and **Hyperbolic Embeddings**. When implementing the Graph Embedding Engine and Student Regressor, double-check all manifold operations (Exponential Map, Logarithmic Map, Distance). Numerical instability near the boundary of the Poincaré Ball () is a known failure mode. Use libraries like `geoopt` or implement "clipping" safeguards.*

---

## 1. Mathematical Foundations (The Geometry)

The semantic space is **Hyperbolic**, specifically the **Poincaré Ball Model** (), to optimally represent hierarchical ontologies (SUMO) with low distortion.

### **1.1 The Manifold**

The manifold is defined as .
The Riemannian metric tensor is , where  and  is the Euclidean metric.

### **1.2 Distance Metric (Loss Function)**

The distance between two vectors  is not Euclidean. The Coding Agent must implement the **Poincaré Distance**:



*Use this as the Loss Function for the Graph Embedding training phase.*

### **1.3 The Projection (The Bridge)**

The Main LLM operates in Euclidean Space. We need a differentiable projection from the Hyperbolic manifold to the tangent space at the origin (Euclidean approximation).
The **Logarithmic Map** at origin () is sufficient for the "Sidecar" injection:



*Note: For numerical stability,  can be approximated or clipped.*

---

## 2. Phase 1: The Offline Harvester (Tier 2 Generation)

**Goal:** Generate the "Induced Graph" by bridging the gap between the static SUMO ontology (Tier 1) and the target corpus.

### **Algorithm 1: N-Gram Extraction & Filtering**

1. **Input:** Raw Text Corpus .
2. **tokenize():** Use a high-recall Noun Phrase chunker (e.g., `spacy` or simple POS patterns).
3. **count():** Calculate term frequency.
4. **Filter Set :** Load all SUMO terms (Tier 1).
5. **Identify Candidates :** .

### **Algorithm 2: LLM-Based Induction (The Agent)**

For each candidate , execute the following Prompt Logic:

* **Context:** Provide a list of top 50 high-level SUMO nodes (e.g., `Process`, `Device`, `Agent`).
* **Prompt:** *"Define the term '' strictly using relationships to the provided SUMO nodes. Output in Tuple Format: `(Subject, Predicate, Object)`."*
* **Example Output for "CUDA Kernel":**
* `('CUDA_Kernel', 'subclass', 'ComputerProgram')`
* `('CUDA_Kernel', 'part_of', 'GPU_Computing')`


* **Action:** Insert new node  and edges into the Master Graph .

---

## 3. Phase 2: Graph Embedding Engine (The Teacher)

**Goal:** Assign a fixed Hyperbolic Vector  to every node  in the Master Graph .

### **Algorithm 3: Poincaré Embedding Training**

*Do not use standard Node2Vec. Use Riemannian Optimization.*

1. **Initialization:** Randomly initialize vectors in  (Uniform distribution within radius 0.001).
2. **Sampling:** For each edge  in , sample  negative nodes .
3. **Loss Function:** Maximize similarity for connected nodes, minimize for unconnected.


4. **Optimizer:** Use **Riemannian Adam (RADAM)**. Standard Adam will push vectors out of the ball.
5. **Output:** A static Dictionary `Map[NodeID] -> Tensor(100)`.

---

## 4. Phase 3: The Student Tokenizer (The Distillation)

**Goal:** Train a fast neural network to map Raw Text  Hyperbolic Vector.

### **4.1 Architecture (The Byte-CNN)**

* **Input:** Sequence of UTF-8 Bytes  (Length , Pad/Truncate).
* **Embedding:** `nn.Embedding(256, 64)` (256 byte values).
* **Encoder:**
* `Conv1d(in=64, out=128, kernel=3, padding=1)` + `ReLU`
* `Conv1d(in=128, out=256, kernel=5, padding=2)` + `ReLU`
* `GlobalMaxPooling` over length dimension.


* **Head:** `Linear(256, 100)` (No activation at end, but see output constraint).
* **Output Constraint:** The output  must be inside the Poincaré Ball. Apply **clipping projection**:
If , then .

### **4.2 Noise Injection Strategy (Robustness)**

During training, apply `augment(text)` with probability 0.3:

* `swap_chars`: "kernel"  "kerenl"
* `drop_char`: "kernel"  "kernl"
* **Target:** The target vector remains the *clean* concept vector `Map[CUDA_Kernel]`.

---

## 5. Phase 4: The Hybrid Fusion Layer (The "Sidecar")

**Goal:** Integrate the Student's Graph Vector into a standard Transformer (e.g., Llama).

### **5.1 The Mathematical Operation**

The input to the first Transformer Block is defined as:

```python
class HybridEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_dim, graph_dim=100):
        super().__init__()
        # Channel A: Standard BPE
        self.bpe_embed = nn.Embedding(vocab_size, hidden_dim)
        
        # Channel B: Graph Projection
        self.graph_proj = nn.Linear(graph_dim, hidden_dim, bias=False)
        
        # Logarithmic Map (from Hyperbolic to Euclidean Tangent Space)
        # Note: Ideally implemented as a pre-processing step or a static function
        
    def forward(self, bpe_ids, graph_vectors):
        # bpe_ids: [Batch, Seq]
        # graph_vectors: [Batch, Seq, Graph_Dim] (Hyperbolic)
        
        # 1. Project Hyperbolic -> Euclidean Tangent Space (Log Map approx)
        norm = torch.norm(graph_vectors, dim=-1, keepdim=True)
        # Avoid div by zero
        tanh_inv = torch.atanh(torch.clamp(norm, max=1-1e-5)) 
        tangent_vecs = (graph_vectors / (norm + 1e-6)) * tanh_inv
        
        # 2. Linear Projection to Model Dimension
        sem_embed = self.graph_proj(tangent_vecs)
        
        # 3. Summation
        token_embed = self.bpe_embed(bpe_ids)
        return token_embed + sem_embed

```

---

## 6. Implementation Checklist for the Agent

1. **Dependency Graph:**
* `networkx`: For graph management.
* `geoopt`: For Riemannian Optimization (RADAM) and Manifold operations.
* `tokenizers`: For the standard BPE channel.
* `pytorch`: Core logic.


2. **Failure Modes to Handle:**
* **NaN in Arccosh:** Occurs if input is exactly 1.0. Implement clamp `min=1+epsilon`.
* **Dimensional Collapse:** If the Student maps everything to zero. Monitor the average norm of predicted vectors.
* **BPE/Graph Misalignment:** The BPE sequence length must match the Graph Tokenizer sequence length. *Constraint:* The Graph Tokenizer should operate on "Words" (split by whitespace), while BPE splits subwords. **Mapping Strategy:** Use `scatter_add` to broadcast the Graph Vector of the word "Kernel" to all its BPE constituents (`["Ker", "nel"]`).


3. **Execution Order:**
1. Run Harvester (Gen Tier 2).
2. Run Graph Embedder (Train ).
3. Train Student Tokenizer (Regress Text  ).
4. Initialize Hybrid Transformer.
5. Train Main LLM.