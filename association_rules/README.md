# Association Rules
To identify frequently co-occurring items in <a href="https://lionbridge.ai/datasets/24-best-ecommerce-retail-datasets-for-machine-learning/">transactional data</a>

<hr>

## Online Interactive Demo:
* <a href="https://danielyang.shinyapps.io/association_rules/">Identifying association rules in transactional datasets</a>

<hr>

## What is an Association Rule?
* { Itemset (Antecedent) } => { Itemset (Consequent) }, or equivalently,
* { Left-Hand Side of the rule} => { Right-Hand Side of the rule }, or equivalently,
* { Rule Body } => { <a href="https://www.ibm.com/support/knowledgecenter/SSEPGG_9.7.0/com.ibm.im.model.doc/c_rule_body_and_rule_head.html">Rule Head</a> }


#### Examples:
* { citrus fruit, root vegetables } => { other vegetables }
* { curd, yogurt } => { whole milk }

<hr>

## Algorithm 1 -- <a href="https://www.geeksforgeeks.org/apriori-algorithm/">Apriori</a>:

Concept | Description | Use | Parameters
--- | --- | --- | ---
<b>support</b> | baseline probability of itemsets,<br/><i>n</i><sub>z</sub>/<i>n</i><sub>total</sub> | to focus on frequent itemsets | minimum support ≥ sigma (e.g., 0.01)
<b>confidence</b> | conditional probability of itemset Y given itemset X,<br/><i>p</i>(<i>E</i><sub>Y</sub> \| <i>E</i><sub>X</sub>) = <i>p</i>(<i>E</i><sub>Y</sub> ∩ <i>E</i><sub>X</sub>) / <i>p</i>(<i>E</i><sub>X</sub>) | to derive rules of {X} -> {Y} | minimum confidence ≥ gamma (e.g., 0.5)
<b>lift</b> | importance of rules as tested against stochastic independence,<br/><i>p</i>(<i>E</i><sub>Y</sub> ∩ <i>E</i><sub>X</sub>) / (<i>p</i>(<i>E</i><sub>X</sub>)\*<i>p</i>(<i>E</i><sub>Y</sub>)) | to compare importance of rules objectively | * lift > 1: {X} increases {Y};<br/>* lift = 1: {X} is independent of {Y};<br/>* lift < 1: {X} decreases {Y}

<hr>

## Algorithm 2 -- <a href="https://www.geeksforgeeks.org/ml-eclat-algorithm/">Eclat</a>:

Whereas the Apriori algorithm works horizontally via the <b>Breadth</b>-First Search of a graph, the ECLAT (Equivalence Class Clustering and bottom-up Lattice Traversal) algorithm works vertically via the <b>Depth</b>-First Search of a graph, making the ECLAT <i>faster</i> than the Apriori algorithm.

<hr>

## Algorithm 3 -- <a href="https://en.wikipedia.org/wiki/Association_rule_learning#FP-growth_algorithm">FP Growth</a>:

FP (Frequent Pattern) growth algorithm is <a href="https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.403.1361&rep=rep1&type=pdf">more efficient</a> than the apriori algorithm. FP growth represents the database in a compressed manner using an FP-tree, and uses a recursive approach to identify the frequent itemsets in a divide-and-conquer manner. See <a href="https://www.quora.com/What-is-the-difference-between-FPgrowth-and-Apriori-algorithms-in-terms-of-results">related article</a> for full comparison between apriori and FP growth algorithms.

<hr>

## Examples:
lhs | => | rhs | support | confidence | lift
--- | --- | --- | --- | --- | ---
{ citrus fruit, root vegetables } | => | { other vegetables } | 0.01037112 | 0.5862069 | 3.029608
{ curd, yogurt } | => | { whole milk } | 0.01006609 | 0.5823529 | 2.279125

<hr>

## Challenges:

Challenge | Description
--- | ---
Rare item problem | rare itemsets not meeting a minimal support threshold (sigma) are ignored.
Longer itemsets problem | long itemsets tend to have lower support.
Confidence ignoring Y | relying on 'confidence' alone may be misleading; instead, one should rely on 'lift'.
Sigma/gamma are user-defined | what is the potential risk of defining these parameters one way or the other?

<hr>

## References:
* R package <a href="https://cran.r-project.org/web/packages/arules/index.html">arules</a> and <a href="https://michael.hahsler.net/SMU/EMIS7331/slides/chap6_basic_association_analysis.pdf">lecture notes</a> by <a href="https://michael.hahsler.net/">Michael Hahsler</a>
* IBM knowledge center: <a href="https://www.ibm.com/support/knowledgecenter/SSEPGG_9.7.0/com.ibm.im.model.doc/c_associations.html">association mining</a>
* <a href="https://www-users.cs.umn.edu/~kumar001/dmbook/index.php">Introduction to Data Mining</a> (Chapters <a href="https://www-users.cs.umn.edu/~kumar001/dmbook/ch5_association_analysis.pdf">5</a> and 6)
* Tools for performing <a href="https://journal.r-project.org/archive/2019/RJ-2019-048/RJ-2019-048.pdf">classification based on association rules</a> (e.g., R packages <a href="https://cran.r-project.org/web/packages/arulesCBA/arulesCBA.pdf">arulesCBA</a> and <a href="https://cran.r-project.org/web/packages/rCBA/index.html">rCBA</a>)
* <a href="https://github.com/AmirAli5/Machine-Learning">Recommendation system for market basket optimisation using apriori algorithm</a>
* A comprehensive R <a href="https://rpubs.com/shah_np/463712">example</a> of analyzing UCI online retail data using apriori algorithm, including various ways of plotting rules
