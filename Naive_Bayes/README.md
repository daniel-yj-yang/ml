
<p align="center"><img src="./images/Bayes_theorem.png" width="700px"><br/><a href="https://en.wikipedia.org/wiki/Bayes%27_theorem">Bayes' theorem<a/></p>

<hr>

Probability | Example | Interpretation
--- | --- | ---
P(Hypothesis \| Event) | P(class="Buying_product_Y" \| behavior="clicking_on_link_A") | A posterior probability 
P(Event \| Hypothesis) | P(behavior="clicking_on_link_A" \| class="Buying_product_Y") | This is from our training data. Among customers who bought product Y, how likely the customer also clicked on a specific link A
P(Hypothesis) | P(class="Buying_product_Y") | The proportion of customers buying product Y (without any knowledge of the links they clicked)
P(Event) | P(behavior="clicking_on_link_A") | The proportion of customers clicking link A (without any knowledge of the product they bought)

```
