## Horchunk

This package contains an implementation of a dynamic sliding-window-based semantic chunking method, with a tunable 
parameter for the cosine similarity threshold, proposed in this 
[research paper](https://journals.uran.ua/eejet/article/view/326177/317250).

The method was primarily inspired by the percentile-based semantic chunking method presented by Greg Kamradt in the 
[5 Levels Of Text Splitting notebook](https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb).

I would also like to acknowledge the work of Brandon Smith and Anton Troynikov for the development of the 
[Chunking Evaluation Framework](https://research.trychroma.com/evaluating-chunking) for RAG systems, which was used 
during the method's development. In addition to the framework, their work presents an overall study of the quality of 
different chunking methods.

### Implemented Method

#### Semantic Splitting 

Initial text is divided firstly in paragraphs with regexp `\n+`, then in sentences based on the delimiters: `!?.`.
After retrieving divided sentences the algorithm logic starts.

The algorithm logic is based on the sliding window technique with a dynamically changing size. 
The algorithm sets the pointer $l$ to the first sentence, the pointer $r$ to the second, and then iteratively moves 
the pointer $r$, expanding the window $W[l,r]$. The sentence at the pointer $l$ and the window $W[l,r]$ are 
transformed into vectors using the embedding model $E$: 

$$
\begin{cases}
v_l = E(S_l), \\
v_w = E\left(W[l, r]\right).
\end{cases}
$$

Then distance $d$ between vectors $v_l$ and $v_w$ is calculated using the cosine similarity formula: 

$$
d = \frac{v_l \cdot v_w}{\|v_l\| \cdot \|v_w\|}.
$$

If distance $d$ is less that the specified threshold $t$: the document $W[l,r)$ is generated. 

Then the $l$ and $r$ pointers are updated as follows: 

$$
\begin{cases}
l = r, \\
r = r + 1.
\end{cases}
$$

The algorithm continues to work until: 

$$
r \leq i, 
$$

where $i$ is the number of sentences. 

The chunking forming logic described above is depicted in figure 1-2.

![fig_1](./images/chunking_1.png)

Fig. 1. Finding the cosine similarity between $S_l$ and the window $W[l,r]$

![fig_2](./images/chunking_2.png)

Fig. 2. Formation of document $W[l,r]$ when passing the set threshold $t=0.85$ by distance $t$

When expanding window $W[l,r]$, it is cler how the semantic meaning in form of vectors obtained using the
embedding model gradually "drifts" from sentence $S_l$ to the sequentially formed windows 
$W[l,r]$, $W[l,r+i]$ (Fig. 3). 

![Semantic drift](./images/semantic_drift.png) 

Fig. 3. Drift of semantic meaning when the window expands

Adding a new sentence moves the window $W[l, r]$ away from sentence $S_l$. If the new sentence $S_r$ is very 
semantically  distant  from  $S_l$, the cosine similarity will immediately move away. If it is close, the similarity 
will not change much.

Using  a  high  threshold  value  of  $t=0.95$, depending on the text, will lead to smaller documents with 1–2 
sentences, while a low threshold of $t=0.72$ – to larger ones. 

For additional control over the documents being created, the $α$ parameter is used. It controls the maximum number of 
sentences in a document to prevent the creation of excessively large documents with semantically similar sentences. 
Such situations can occur when processing lists or tables with similar values.

For more details, check the [research paper](https://journals.uran.ua/eejet/article/view/326177/317250). 

#### Binary Search Tuning 

### Benchmarks

### Installation

To install the package, run: 

```bash
pip install "git+https://github.com/panalexeu/horchunk.git"
```

### Usage 
