
<p align="center"><img src="./images/Bayes_theorem.png" width="700px"><br/><a href="https://en.wikipedia.org/wiki/Bayes%27_theorem">Bayes' theorem<a/></p>

<hr>

Probability | Example | Interpretation
--- | --- | ---
P(Hypothesis \| Event): A posterior probability | P(class="Buying_product_Y" \| behavior="clicking_on_link_A") | ---
P(Event \| Hypothesis) | P(behavior="clicking_on_link_A" \| class="Buying_product_Y") | This is from our training data. Among customer bought product Y, how likely the customer also clicked on a specific link A
P(Hypothesis) | P(class="Buying_product_Y") | The proportion of customers buying product Y (without any knowledge of the links they clicked)
P(Event) | P(behavior="clicking_on_link_A") | The proportion of customers clicking link A (without any knowledge of the product they bought)

is the probability of an e-mail containing the word sex. This is simply the proportion of e-mails containing the word sex in our entire training set. We divide by this value because the more exclusive the word sex is, the more important is the context in which it appears. Thus, if this number is low (the word appears very rarely), it can be a great indicator that in the cases it does appear, it is a relevant feature to analyze.
```
