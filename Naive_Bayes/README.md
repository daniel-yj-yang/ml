# Naive Bayes 

<hr>

## Example

<hr>

## Concept

### 1. <a href="https://en.wikipedia.org/wiki/Bayes%27_theorem">Bayes' theorem</a>

<p align="center"><img src="./images/Bayes_theorem.png" width="700px"></p>

<hr>

#### Example of Bayes' theorem

<p align="center">p(H | E) = p(E | H) * p(H) / p(E)</p>

Probability | Example | Interpretation
--- | --- | ---
P(Hypothesis\|Event) | P(class="Buying_product_Y" \| behavior="clicking_on_link_A") | This is the posterior probability we want to estimate. For example, among customers who have clicked on a specific link #A, the proportion of them who then also bought product #Y
P(Event\|Hypothesis) | P(behavior="clicking_on_link_A" \| class="Buying_product_Y") | This is from the training data. Among customers who bought product #Y, the proportion of them who have also clicked on a specific link #A beforehand
P(Hypothesis) | P(class="Buying_product_Y") | The proportion of customers who bought product #Y (without any knowledge of the links they have clicked beforehand)
P(Event) | P(behavior="clicking_on_link_A") | The proportion of customers who clicked link #A (without any knowledge of the product they then bought)

<hr>

### 2. Naive

It assumes that every pair of features being classified is independent of each other
