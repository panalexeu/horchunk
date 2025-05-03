## Horchunk

This package contains an implementation of a dynamic sliding-window-based semantic chunking method, with a tunable 
parameter for the cosine similarity threshold, proposed in this 
[research paper](https://journals.uran.ua/eejet/article/view/326177/317250).

The method was primarily inspired by the percentile-based semantic chunking method presented by Greg Kamradt in the 
[5 Levels Of Text Splitting notebook](https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb).

I would also like to acknowledge the work of Brandon Smith and Anton Troynikov for the development of the 
[Chunking Evaluation Framewor](https://research.trychroma.com/evaluating-chunking) for RAG systems, which was used 
during the method's development. In addition to the framework, their work presents an overall study of the quality of 
different chunking methods.

### Benchmarks

The benchmarking was conducted using the framework proposed in the [Chunking Evaluation Framework](https://research.trychroma.com/evaluating-chunking). Part of the results for other splitting algorithms were also taken from this work.

The embedding model used is **all-MiniLM-L6-v2**, and the number of retrieved chunks was set to 5.

The results for the developed dynamic sliding-window-based semantic chunking method are marked with asterisks.

| Splitting algorithm | Parameters                    | Recall          | Precision      | Precision Ω     | IoU            |
| ------------------- | ----------------------------- | --------------- | -------------- | --------------- | -------------- |
| Recursive           | size: 250 t.; overlap: 125 t. | 78.7 ± 39.2     | 4.9 ± 4.5      | 21.9 ± 14.9     | 4.9 ± 4.4      |
| Token Text          | size: 250 t.; overlap: 125 t. | **82.4 ± 36.2** | 3.6 ± 3.1      | 11.4 ± 6.6      | 3.5 ± 3.1      |
| Recursive           | size: 250 t.; overlap: 0 t.   | 78.5 ± 39.5     | 5.4 ± 4.9      | 26.7 ± 18.3     | 5.4 ± 4.9      |
| Token Text          | size: 250 t.; overlap: 0 t.   | 77.1 ± 39.3     | 3.3 ± 3.0      | 16.4 ± 10.3     | 3.3 ± 3.0      | 
| Recursive           | size: 200 t.; overlap: 0 t.   | 75.7 ± 40.7     | 6.5 ± 6.2      | 31.2 ± 18.4     | 6.5 ± 6.1      | 
| Token Text          | size: 200 t.; overlap: 0 t.   | 76.6 ± 38.8     | 4.1 ± 3.7      | 19.1 ± 11.0     | 4.1 ± 3.6      |
| Kamradt Mod.        | size: 250 t.; overlap: 0 t.   | 63.1 ±  46.9    | 2.7 ± 3.8      | 13.5 ± 13.3     | 2.7 ± 3.8      | 
| Kamradt Mod.        | size: 200 t.; overlap: 0 t.   | 67.9 ± 44.9     | 3.5 ± 4.1      | 16.0 ± 14.9     | 3.5 ± 4.1      | 
| Cluster             | size: 250 t.; overlap: 0 t.   | 77.3 ± 38.6     | 6.1 ± 5.1      | 28.6 ± 16.7     | 6.0 ± 5.1      | 
| Cluster             | size: 200 t.; overlap: 0 t.   | 75.2 ± 39.9     | 7.2 ± 6.1      | 33.6 ± 20.0     | 7.2 ± 6.0      |
| Dyn. Window*        | $t$: 0.72; $α$: 6.            | 74.9 ± 40.0     | 7.6 ± 6.5      | 35.0 ± 18.3     | 7.4 ± 6.3      |   
| Dyn. Window*        | $t$: 0.72; $α$: 3.            | 68.7 ± 41.3     | **10.3 ± 8.5** | **48.4 ± 19.8** | **10.0 ± 8.3** |    

### Implemented Method

#### Semantic Splitting 

Initial text is first divided into paragraphs using the regular expression `\n+`, then into sentences based on the delimiters: `!?.`.  
After splitting the text into sentences, the core logic of the algorithm begins.

The algorithm is based on the sliding window technique with a dynamically changing size.  It sets the pointer $l$ to the
first sentence, the pointer $r$ to the second, and then iteratively moves the pointer $r$, expanding the window $W[l,r]$. 
The sentence at pointer $l$ and the window $W[l,r]$ are transformed into vectors using the embedding model $E$: 

$$
\begin{cases}
v_l = E(S_l), \\
v_w = E\left(W[l, r]\right).
\end{cases}
$$

Then the distance $d$ between vectors $v_l$ and $v_w$ is calculated using the cosine similarity formula: 

$$
d = \frac{v_l \cdot v_w}{\|v_l\| \cdot \|v_w\|}.
$$

If the distance $d$ is less than the specified threshold $t$, the document $W[l,r)$ is generated. 

Then the $l$ and $r$ pointers are updated as follows: 

$$
\begin{cases}
l = r, \\
r = r + 1.
\end{cases}
$$

The algorithm continues until: 

$$
r \leq i, 
$$

where $i$ is the number of sentences. 

The chunking logic described above is depicted in Figures 1–2.

![fig_1](./images/chunking_1.png)

Fig. 1. Finding the cosine similarity between $S_l$ and the window $W[l,r]$

![fig_2](./images/chunking_2.png)

Fig. 2. Formation of document $W[l,r]$ when exceeding the threshold $t = 0.85$

When expanding the window $W[l,r]$, it is clear how the semantic meaning - in the form of vectors obtained using the 
embedding model - gradually "drifts" from sentence $S_l$ to the sequentially formed windows $W[l,r]$, $W[l,r+i]$ (Fig. 3). 

![semantic_drift](./images/semantic_drift.png) 

Fig. 3. Drift of semantic meaning as the window expands

Adding a new sentence moves the window $W[l, r]$ further from sentence $S_l$. If the new sentence $S_r$ is very 
semantically distant from $S_l$, the cosine similarity will decrease significantly. If it is close, the similarity 
will change only slightly.

Using a high threshold value of $t = 0.95$, depending on the text, will lead to smaller documents with 1–2 sentences, 
while a low threshold of $t = 0.72$ will result in larger ones. 

For additional control over the documents being created, the $α$ parameter is used. It limits the maximum number of
sentences in a document to prevent the creation of excessively large documents with semantically similar sentences.
Such situations can occur when processing lists or tables with similar values.

For more details, check the [research paper](https://journals.uran.ua/eejet/article/view/326177/317250).

#### Binary Search Threshold Tuning 

The developed adjustment algorithm is based on the binary search algorithm. The source text is divided into paragraphs 
and then into sentences, just like at the start of the `Semantic Splitting` algorithm.

Next, based on the specified size of the static window $m$, the divided sentences are sequentially processed and chunks of
text are formed from $m$ sentences:

$$
C_i = \{s_t, s_{t+1}, \ldots, s_{t+m-1}\}.
$$

For each obtained chunk $C_i$, the first sentence $C_l$ and the entire chunk $C_i$ are transformed into vectors
using the embedding model $E$:

$$
\begin{cases}
v_l = E(C_l), \\
v_c = E(C_i).
\end{cases}
$$

Next, the distance $d_i$ between $v_l$ and $v_c$ is computed using cosine similarity.

A dictionary is formed from the obtained distances and chunks:

$$
D = \{(d_1, C_1), (d_2, C_2), \ldots, (d_i, C_i)\}.
$$

The created dictionary $D$ is sorted in ascending order based on the distances $d_i$. 

After receiving the sorted dictionary $D$, the binary search algorithm is launched with human evaluation.
The evaluation is performed by entering a command in the terminal to increase or decrease the threshold value.
If the generated chunk with a given distance is semantically complete in the human's opinion, the threshold is decreased.
If the generated chunk is semantically different, it is increased. After the evaluation is complete, the identified 
distance is returned, rounded to two decimal places. This final value is taken as the threshold $t$ (Fig. 4).

![tuning](./images/tuning.png)

Fig. 4. The process of finding the threshold $t$ using human evaluation.

Using binary search-based tuning, the threshold value of the cosine similarity can be adjusted based on the data 
to be split. In addition, since the static window size of $m$ sentences is set during the formation of dictionary $D$, 
the tuning process finds the minimum threshold value for forming semantically complete documents of $m$ sentences or more. 

For more details, check the [research paper](https://journals.uran.ua/eejet/article/view/326177/317250).

### Installation

To install the package, run: 

```bash
pip install "git+https://github.com/panalexeu/horchunk.git"
```

### Usage 
