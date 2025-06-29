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

where $n$ is the number data points, $y$ is the $y$ value of the data point (what we may call the "true" value later on), and $m, b$ are the slope and intercept terms in the classic definition of a line ($y = mx + b$) respectively. 

*Note: The added factor of $\frac{1}{2n}$ is there simply to make differentiation down the line neater. Though it may change what the actual calculated loss is at a given point, it does* not *change where minimum of the function lays, which is what we are interested in.*

Semantically, this function asks, "how well does my line $mx + b$ model the data?" Of course, a lower value means that line models the data better. Hence, for the best model, we must minimize the value of $L(m, b)$.

We can do so by solving for where $\nabla L(m, b) = (0, 0)$. This will require us to first find $\frac{\partial L}{\partial m}$ and $\frac{\partial L}{\partial b}$. We start with $\frac{\partial L}{\partial m}$.

$$
\begin{aligned}
\huge \frac{\partial L}{\partial m} &\huge= \huge \frac{\partial}{\partial m} (\frac{1}{2n} \sum_{i=1}^{n} (y - (mx_i + b))^2) \\
&\huge = \frac{1}{2n} \sum_{i=1}^{n} \frac{\partial}{\partial m} (y - (mx_i + b))^2 \\
&\huge = \frac{1}{2n} \sum_{i=1}^{n} 2(y - (mx_i + b))(-x_i) \\
&\huge = -\frac{1}{2n} \sum_{i=1}^{n} x_i(y - (mx_i + b))
\end{aligned}
$$

