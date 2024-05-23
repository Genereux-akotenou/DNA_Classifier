# DNA Sequence Classification Approaches

## Approach 1: Single Nucleotide as a Token

### Description
- Treat each base (A, T, C, G) as a single token.
- Encode each base numerically (e.g., A=0, T=1, C=2, G=3).
- Train a model on the sequence of encoded bases.

### Pros
- Simple and straightforward.
- Preserves the positional information of each base.

### Cons
- Limited contextual information.
- May not capture long-range dependencies well.

### Model
- **RNNs/LSTMs**: Good for capturing sequential dependencies.
- **CNNs**: Effective for capturing local patterns and motifs in sequences.
- **Transformers**: Excellent for capturing long-range dependencies and complex patterns with attention mechanisms.

### Todo
- [Click to Open Notebook 1](./02-approach1_single_nucleotide_position.ipynb)

## Approach 2: k-mer Representation with Frequency Analysis 

### Description
- Break the DNA sequence into k-mers (subsequences of length k).
- Perform frequency analysis to create a feature vector based on the occurrence of each k-mer.
- Use this feature vector as input to the model.

### Pros
- Captures local context within each k-mer.
- Simplifies the input representation by reducing it to frequency counts.

### Cons
- Loses positional information beyond the k-mer length.
- Treating it as a [bag-of-words](https://en.wikipedia.org/wiki/Bag-of-words_model) model ignores the order of k-mers.

### Model
- **RandomForest**
- **GaussianProcess**
- **....**

### Todo
- [Click to Open Notebook 2](./03-approach2_kmer_frequence.ipynb)

## Approach 3: k-mer Representation with Position Analysis (DEEPLEARNING MODEL)

### Description
- Break the DNA sequence into k-mers.
- Use embeddings to represent each k-mer, capturing more complex relationships and context.
- Train a model using these embeddings.

### Pros
- Captures richer contextual information through embeddings.
- Can capture long-range dependencies if using advanced embedding techniques (e.g., BERT).

### Cons
- More complex and computationally intensive.
- Requires large amounts of data to train effective embeddings.

### Model
- **RNNs/LSTMs**: Good for capturing sequential dependencies.
- **CNNs**: Effective for capturing local patterns and motifs in sequences.
- **Transformers**: Excellent for capturing long-range dependencies and complex patterns with attention mechanisms.

### Todo
- [Click to Open Notebook 3 (FROM SCRATCH)](./04-approach3_kmer_position(from_scratch).ipynb)
- [Click to Open Notebook 4 (TRANSFER LEARNING)](./05-approach3_kmer_position(transfert_learning).ipynb)
