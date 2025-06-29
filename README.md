# Homemade Linear Regressor

This project marks the beginning of a journey towards a goal I have set for myself: **Gain a deep understanding of the theory behind the several foundational machine learning models by building them from scratch.** I've always wanted to intimately understand all the math that goes into these imprssive algorithms and I think this is an exciting way to do it. The rest of this README, as well as those for my other  homemade models, is nothing novel and more just a place for me to document everything I've learned.

## Motivation

The idea of linear regression dates back to more than 2 centuries ago. A cornerstone of statistics, this model elegant, fast, and easily interpretable, making it a great first choice for exploratory data analysis.

## Mathematical Foundations

### Univariate Case

#### Loss Function

There are several loss functions one may choose to optimize when fitting their model. Here we choose the **sum of the squared residuals.** To breifly explain this loss function, suppose we have some 2-dimensional data i.e. a set of $(x, y)$ coordinates, and any line through that data. For each data point, this function will take the difference between the point's $y$ value and the value of the line at the point's $x$ value. Squaring this difference gives us the *squared residual* for that point. The sum of all the squared residuals is the calculated loss. 

#### Minimizing Loss

Since we are only concerned with the univariate case at the moment, we can define our loss function like this:

$$ 
\huge L(m, b) = \frac{1}{2n} \sum_{i=1}^{n} (y - (mx_i + b))^2
$$

