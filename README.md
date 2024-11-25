Here's the revised README without video references, tailored for Bindwell and including placeholders for PyMOL-generated images:

---

# **Simple, Hackable Protein Transformers for Bindwell**

This repository provides a **simple and hackable implementation of protein transformer models** for education, research, and real-world applications. It reflects **Bindwell's commitment to democratizing AI tools for protein-ligand interactions** and advancing sustainable agriculture.

---

## **Repository Features**

This project is built for clarity, simplicity, and flexibility. It includes:

1. **Protein Data Loaders and Tokenizers:** 
   A straightforward approach to process protein sequences for modeling.
2. **Decoder-Only Transformer Models:**
   A fully spelled-out implementation for easy experimentation.
3. **Training and Evaluation Loops:** 
   Simple, customizable loops for effective model training and assessment.

---

## **Getting Started**

### **Protein Data Loaders**

Our `ProteinDataset` class supports character-level tokenization (common in protein language models). Example usage:

```python
proteins = [
    "MCLLSLAAATVAARRTPLRLLGRGLAAAMSTAGPLKSV",
    "MSSQIKKSKTTTKKLVKSAPKSVPNAAADDQIFCCQFE",
    "MCLLSLAAATVAARRTPLRLLGRGLAAAMSTAGPLKSV",
]

chars = "ACDEFGHIKLMNPQRSTVWY"
max_length = 38

dataset = ProteinDataset(proteins, chars, max_length)
```

---

### **Transformer Model**

The model follows a "decoder-only" transformer architecture, where a triangular mask predicts the next token. Define a transformer configuration:

```python
config = ModelConfig(vocab_size=20, block_size=128,
                     n_layer=4, n_head=4, embed_dim=64)

model = Transformer(config)
```

---

### **Training the Model**

Customize the training loop with command-line arguments. Create an optimizer, initialize the model, and train using backpropagation. The pipeline evaluates the loss periodically and generates protein samples for visualization.

**Protein Folding Visualization:**  
To track the structural predictions during training, the workflow integrates **ESMFold** for 3D visualization. Below is an example generated protein and its folded structure:

```bash
python main.py -i acyp.fa -o acyp
```

**Generated Sample Sequence:**

```
>sample
MAREVKHLVIYGRVQGVGYRAWAENEAMGLGLEGWVRNRRDGSVEALVVGGPADVSAMITRCRHGPPTAGIVSLLEETCPDAGIPPSRGFKQLPTV
```

### **Visualizing Protein Structures**

Generated proteins can be visualized using **PyMOL**. Below is an example comparing the generated protein structure (purple) to a homolog crystal structure (gray, PDB: 1URR):

![Generated Protein Structure](img/example-folded.png)  
*Generated protein folded using ESMFold.*

![Superposition with Crystal Structure](img/example-folded-crystal.png)  
*Superposition of generated protein (purple) and crystal structure (gray).*

---

## **Data Clustering for Better Performance**

Cluster datasets using **MMseqs2** to improve model performance. For example, cluster protein sequences to 90% sequence identity:

```bash
mmseqs createdb acyp.fa DB
mmseqs cluster -c 0.95 --min-seq-id 0.90 DB clust90 tmp
mmseqs createsubdb clust90 DB rep90
mmseqs convert2fasta rep90 rep90.fa
```

This clustering balances sequence diversity and training efficiency.

---

## **Installation**

Set up the environment with minimal dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch rich biotite plotly tensorboard
```

---

## **Key Takeaways**

- **Hackable Design:** Simplified transformer architecture, easy for experimentation.
- **Real-World Impact:** Useful for predicting protein-ligand interactions, advancing pesticide discovery, and driving sustainable farming.
- **Visual Feedback:** Integration with ESMFold ensures rapid 3D structure validation during model development.

---

**Bindwell**: Revolutionizing AI for sustainable agriculture. 🌱  

--- 
