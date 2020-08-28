# <a href="https://en.wikipedia.org/wiki/Recommender_system">Recommendation System</a>
To recommend what a user would like to the user so as to help users quickly find compelling content in a large collection of items.

Disclaimer: this repository is based on my own research using various knowledge bases, including <a href="https://developers.google.com/machine-learning/recommendation">Google's machine learning course</a>.

<hr>

## Example

Please see <a href="../collaborative_filtering">collaborative filtering</a>

<hr>

## Approaches

Approach | Notes
--- | ---
<a href="../content-based_filtering">Content-based filtering</a> | - The model makes user-specific recommendation <b>without</b> using any information from other users.
<a href="../collaborative_filtering">Collaborative filtering</a> | - User-based similarity<br/>- Item-based similarity<br/>- The model uses these two kinds of similarities <b>simultaneously</b> to provide recommendations.<br/>- Can address some limitations in <a href="../content-based_filtering">content-based filtering</a>
<a href="../DNN-softmax">Deep neural network: Softmax</a> | - Can address some limitation in the matrix factorization approach in <a href="../collaborative_filtering">collaborative_filtering</a> 
Hybrid | - Combining collaborative filtering, content-based filtering, and others.<br/><br/>- The 2007 winner of the Netflix prize said the following:<br/><br/> * "Our experience is that most efforts should be concentrated in deriving substantially different approaches, rather than refining a single technique.";<br/> * "A blend of #8, #38, and #92, with weights 0.1893, 0.4225, and 0.4441, respectively, would already achieve a solution with an RMSE of 0.8793."

<hr>

## Key Concepts

Term | ---
--- | ---
Query | The information a system uses to make recommendations. Also known as contexts.<br/>For example:<br/>- User information: user ID, previously interacted items<br/>- Other context: user device, time of day
Item | The entity a system recommends. Also known as documents.
<a href="https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture">Embedding</a> | - A <b>mapping</b> from the queries and items to an embedding space;<br/>- <a href="http://projector.tensorflow.org/">Example</a> by tensorflow<br/>- It is the individual element of the user-item interaction matrix.<br/>- It is the lower-dimensional internal compact representation (like hidden layer) of the higher-dimensional input.
Embedding space | - <b><i>E</i> = <a href="https://en.wikipedia.org/wiki/Real_number">&reals;</a><sup>n</sup></b>, a vector with n real-valued entries (n-dimension).<br/>- Similar items (such as items frequently purchased by the same user) are close together in the embedding space.<br/>- It is the embedding matrix, and the internal compact representation space.
Similarity measure | - A function, a <a href="https://en.wikipedia.org/wiki/Linear_map">linear map</a> between two vector spaces (<i>E</i> x <i>E</i> and &reals;), s: <i>E</i> x <i>E</i> &rarr; &reals;, returns a scalar of the similarity of a pair of embeddings.<br/>- Example: s(q, x) is a similarity measure between [a query embedding q ∈ E] and [an item embedding x ∈ E]<br/><table><tr><th>measure</th><th>s(q, x)</th><th>Notes</th></tr><tr><td>Cosine</td><td>cos(q, x)</td><td>Insensitive to the norm of the embedding</td></tr><tr><td>Dot product</td><td>< q, x ></td><td>Sensitive to the norm of the embedding,<br/>such as popular/frequent items</td></tr><tr><td>(-) Euclidean distance</td><td>ǁ q - x ǁ</td><td></td></tr><tr><td>Custom</td><td>ǁqǁ<sup>α</sup> ǁxǁ<sup>α</sup> cos(q,x), for some α ∈ (0,1)</td><td>Reducing emphasis of the norm of the item</td></tr></table>

<hr>

## <a href="https://developers.google.com/machine-learning/recommendation/overview/types">Stages of the system</a>

Stage | Procedure
--- | ---
1.&nbsp;<a href="https://developers.google.com/machine-learning/recommendation/overview/candidate-generation">Candidate generation</a> | The system quickly nominates a much smaller subset of candidate items from a huge corpus/collection.
2.&nbsp;Scoring and Re-ranking | - From the subset in stage#1, the system more precisely scores and ranks the candidate items, while considering additioanl queries.<br/>- Finally, from an even smaller subset, the system performs a final ranking by considering additinoal constraints, such as items of fresher content.

<hr>

## Stage 1. Candidate Generation

Source | Details
--- | ---
<a href="../DNN-softmax">Deep neural network: Softmax</a> | - Q: Given a user (query embedding <i>q</i>), how to <a href="https://developers.google.com/machine-learning/recommendation/dnn/retrieval">retrieve</a> the relevant items to recommend?<br/>- A: Find the top k items (with respect to <i>V<sub>j</sub></i>) that shows the highest similarity with <i>q</i>, namely, the highest few scores of <<i>q</i>, <i>V<sub>j</sub></i>>
Matrix factorization | Look up the static query (or user) embedding in the user embedding matrix
User features | Related to personalization (e..g, using the deep neural network: softmax)
"Local" vs. "distant" items | Considering geographic or language info
Frequent items | Popular/trending items
Social graph | Items liked or recommended by friends

<hr>

## Stage 2. <a href="https://developers.google.com/machine-learning/recommendation/dnn/scoring">Scoring</a> and <a href="https://developers.google.com/machine-learning/recommendation/dnn/re-ranking">Re-ranking</a>

### Scoring:
Combining candidates from multiple sources of candidate generators, and then using specific criteria to score them to derive the probability of a user interacting with each of them.

### Re-ranking:
Considering additional criteria or constraints.

Additional criterion | Details
--- | ---
Freshness | To incorporate the latest usage information to keep the recommendations up-to-date
Diversity | To avoid boredom
Fairness | To avoid unconscious biases from the training data

<hr>

## References

- Bell et al. (2007) <a href="./references/ProgressPrize2007_KorBell.pdf">The BellKor solution to the Netflix Prize.</a> (downloaded from <a href="https://www.netflixprize.com/assets/ProgressPrize2007_KorBell.pdf">netflixprize.com</a>)
- <a href="https://developers.google.com/machine-learning/recommendation">Google course of recommendation system</a>
