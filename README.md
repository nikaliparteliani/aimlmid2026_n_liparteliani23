# midterm exam

# Calculate the correlation for the given data.
<img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/80a4fe14-e74e-425a-81bb-ecc6f8a864b5" />

## Pearson correlation (r) calculation

### Data points (n = 9)

(-8, -7)
(-6, -5)
(-4, -2)
(-2, 0.7)
(-0.5, 1)
(2, 3)
(4, 4)
(6, 6)
(8, 8)


---

### Step 1 — Means

x̄ = (−8 −6 −4 −2 −0.5 +2 +4 +6 +8) / 9
x̄ = −0.5 / 9
x̄ = −0.0556

ȳ = (−7 −5 −2 +0.7 +1 +3 +4 +6 +8) / 9
ȳ = 8.7 / 9
ȳ = 0.9667


---

### Step 2 — Deviations table

| i | xi | yi | xi−x̄ | yi−ȳ | (xi−x̄)(yi−ȳ) | (xi−x̄)² | (yi−ȳ)² |
|---|----|----|------|------|---------------|----------|----------|
|1|-8|-7|-7.9444|-7.9667|63.30|63.11|63.47|
|2|-6|-5|-5.9444|-5.9667|35.46|35.33|35.60|
|3|-4|-2|-3.9444|-2.9667|11.70|15.56|8.80|
|4|-2|0.7|-1.9444|-0.2667|0.52|3.78|0.07|
|5|-0.5|1|-0.4444|0.0333|-0.01|0.20|0.00|
|6|2|3|2.0556|2.0333|4.18|4.23|4.13|
|7|4|4|4.0556|3.0333|12.31|16.45|9.20|
|8|6|6|6.0556|5.0333|30.47|36.67|25.33|
|9|8|8|8.0556|7.0333|56.67|64.89|49.47|

---

### Step 3 — Sums

Σ(xi − x̄)(yi − ȳ) = 214.60
Σ(xi − x̄)² = 240.22
Σ(yi − ȳ)² = 196.07

---

### Step 4 — Pearson correlation coefficient

r = Σ(xi − x̄)(yi − ȳ) / sqrt[ Σ(xi − x̄)² · Σ(yi − ȳ)² ]

r = 214.60 / sqrt(240.22 × 196.07)
r = 214.60 / 216.99
r ≈ 0.989
---
### ✅ Final result
### Pearson correlation in Python

```python
import numpy as np

x = np.array([-8, -6, -4, -2, -0.5, 2, 4, 6, 8])
y = np.array([-7, -5, -2, 0.7, 1, 3, 4, 6, 8])

r = np.corrcoef(x, y)[0, 1]
print("r =", r)
```
r = 0.9887186998035012



## Spam Classifier – Training and Evaluation

### Environment setup

Create and activate a virtual environment, then install required dependencies:

```bash
nano spam_classifier.py
python -m venv venv
source venv/bin/activate
pip install pandas scikit-learn joblib matplotlib
Model training
Train the spam classifier using the provided CSV dataset and save the trained model:

bash

python spam_classifier.py train --csv n_liparteliani23_48213.csv --out model.joblib
Output:


Saved model to: model.joblib

Intercept: -8.821537266184123
Coef(words) = 0.006640196452056835
Coef(links) = 0.8133056379063894
Coef(capital_words) = 0.40498488879274697
Coef(spam_word_count) = 0.7519649509922152
Explanation:

Intercept – bias term of the logistic regression model

Coef(words) – impact of total word count

Coef(links) – strong indicator of spam (URLs)

Coef(capital_words) – influence of excessive capitalization

Coef(spam_word_count) – frequency of known spam-related words

Model evaluation
Confusion matrix and accuracy obtained during training:


Confusion Matrix:
[[366  13]
 [ 18 353]]

Accuracy: 0.9587
Interpretation:

The model correctly classifies ~95.9% of messages

Low false positives and false negatives indicate good performance

Prediction example
Run inference on a custom input message:

python spam_classifier.py predict --model model.joblib --text "FREE PRIZEls! click http://bad-site.com NOW"

Output:
Extracted features:
{'words': 8, 'links': 1, 'capital_words': 2, 'spam_word_count': 3}

Prediction: LEGIT
Spam probability: 0.00747
Explanation:

The extracted features are passed to the trained model

Despite spam-like keywords, overall probability is low

Final classification: LEGIT





