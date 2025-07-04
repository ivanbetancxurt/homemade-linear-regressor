# Homemade Linear Regressor

This project marks the beginning of a journey towards a goal I have set for myself: **Gain a deep understanding of the theory behind the several foundational machine learning models by building them from scratch.** I've always wanted to intimately understand all the math that goes into these imprssive algorithms and I think this is an exciting way to do it. The rest of this README, as well as those for my other  homemade models, is nothing novel and more just a place for me to document everything I've learned.

## Motivation

The idea of linear regression dates back to more than 2 centuries ago. A cornerstone of statistics, this model is elegant, fast, and easily interpretable, making it a great first choice for exploratory data analysis.

## Mathematical Foundations

### Univariate Case

#### Loss Function

There are several loss functions one may choose to optimize when fitting their model. Here we choose **mean squared error.** To breifly explain this loss function, suppose we have some 2-dimensional data i.e. a set of $(x, y)$ coordinates, and any line through that data. For each data point, this function will take the difference between the point's $y$ value and the value of the line at the point's $x$ value. Squaring this difference gives us the *squared error* for that point. The mean of all the squared errors is the calculated loss. 

#### Minimizing Loss

In the univariate case, we can define our loss function like this:

$$ 
\huge L(m, b) = \frac{1}{2n} \sum_{i=1}^{n} (y_i - (mx_i + b))^2
$$

where $n$ is the number data points, $y_i$ is the $y$ value of the $i$ th data point (the "true" value), and $m, b$ are the slope and intercept terms in the classic definition of a line ($y = mx + b$) respectively. 

*Note: The added factor of* $\frac{1}{2}$ *is there simply to make differentiation down the line neater. Though it may change what the actual calculated loss is at a given point, it does* not *change where minimum of the function lays, which is what we are interested in.*

Semantically, this function asks, "how well does my line $mx_i + b$ model the data?" Of course, a lower value means that line models the data better. Hence, for the best model, we must minimize the value of $L(m, b)$.

We can do so by solving for where $\nabla L(m, b) = (0, 0)$. This requires us to first find $\frac{\partial L}{\partial m}$ and $\frac{\partial L}{\partial b}$.

$$
\begin{aligned}
\huge \frac{\partial L}{\partial m} &\huge= \frac{\partial}{\partial m} (\frac{1}{2n} \sum_{i=1}^{n} (y_i - (mx_i + b))^2) \\
&\huge = \frac{1}{2n} \sum_{i=1}^{n} \frac{\partial}{\partial m} (y_i - (mx_i + b))^2 \\
&\huge = \frac{1}{2n} \sum_{i=1}^{n} 2(y_i - (mx_i + b))(-x_i) \\
&\huge = -\frac{1}{n} \sum_{i=1}^{n} x_i(y_i - (mx_i + b)).
\end{aligned}
$$

$$
\begin{aligned}
\huge \frac{\partial L}{\partial b} &\huge= \frac{\partial}{\partial b} (\frac{1}{2n} \sum_{i=1}^{n} (y_i - (mx_i + b))^2) \\
&\huge = \frac{1}{2n} \sum_{i=1}^{n} \frac{\partial}{\partial b} (y_i - (mx_i + b))^2 \\
&\huge = \frac{1}{2n} \sum_{i=1}^{n} 2(y_i - (mx_i + b))(-1) \\
&\huge = -\frac{1}{n} \sum_{i=1}^{n} y_i - mx_i - b.
\end{aligned}
$$

All that's left to do is set these derivative functions equal to 0 and solve the system of equations. Then we will have found the values for $m$ and $b$ that satisfy $\nabla L(m, b) = (0, 0)$. Note that

$$
\begin{aligned}
&\huge \frac{\partial L}{\partial m} = 0 \\
\huge \Rightarrow &\huge -\frac{1}{n} \sum_{i=1}^{n} x_i(y_i - (mx_i + b)) = 0 \\
\huge \Rightarrow &\huge \sum_{i=1}^{n} x_iy_i - mx_i^2 - bx_i = 0 \\
\huge \Rightarrow &\huge \sum_{i=1}^{n} mx_i^2 + \sum_{i=1}^{n} bx_i = \sum_{i=1}^{n} x_iy_i \\
\end{aligned}
$$

and similarly,

$$
\begin{aligned}
&\huge \frac{\partial L}{\partial b} = 0 \\
\huge \Rightarrow &\huge -\frac{1}{n} \sum_{i=1}^{n} y_i - mx_i - b = 0 \\
\huge \Rightarrow &\huge \sum_{i=1}^{n} y_i - mx_i - b = 0 \\
\huge \Rightarrow &\huge (\sum_{i=1}^{n} mx_i) + nb = \sum_{i=1}^{n} y_i. \\
\end{aligned}
$$

We can use matrices to solve the system. Let

$$
\large A = \begin{bmatrix} \sum x_i^2 & \sum x_i \\\ \sum x_i & n \end{bmatrix}, \\
\vec{c} = \begin{bmatrix} \sum x_iy_i \\\ \sum y_i \end{bmatrix}.
$$

We assume $A$ is invertible. Observe that

$$
\begin{aligned}
&\huge A \begin{bmatrix} m \\\ b \end{bmatrix} = \vec{c} \\
\huge \Rightarrow &\huge A^{-1} (A \begin{bmatrix} m \\\ b \end{bmatrix}) = A^{-1} \vec{c} \\
\huge \Rightarrow &\huge \begin{bmatrix} m \\\ b \end{bmatrix} = A^{-1} \vec{c} \\
\huge \Rightarrow &\huge \begin{bmatrix} m \\\ b \end{bmatrix} = \frac{1}{n \sum x_i^2 - (\sum x_i)^2} \begin{bmatrix} n & -\sum x_i \\\ -\sum x_i & \sum x_i^2 \end{bmatrix} \begin{bmatrix} \sum x_iy_i \\\ \sum y_i \end{bmatrix} \\
\huge \Rightarrow &\huge m = \frac{\sum x_iy_i - n\bar{x}\bar{y}}{\sum x_i^2 - n\bar{x}^2}, b = \bar{y} - m\bar{x}
\end{aligned}
$$

where $\bar{x} = \frac{1}{n} \sum x_i$ and $\bar{y} = \frac{1}{n} \sum y_i$ are the means of all ${x_i}$ and ${y_i}$ respectively.

We now have the values for $m$ and $b$ that minimize our loss function and can now use them on a new univarite sample $x$, like this: $\hat{y} = mx + b$.

### Multivariate Case

The multivariate case is essentially the same as the univariate case, just generalized. In this case we will consider samples of $d$ features.

#### Minimizing Loss

We use the same loss function but modify it to fit our multi-dimensional data points:

$$
\huge L(\vec{w}) = \frac{1}{2n} \sum_{i=1}^{n} (y_i - \tilde{x_i}^T\vec{w})^2
$$

where $\vec{w}$ is a $(d+1 \times 1)$ column vector holding a bias term and the learned weights for each of the $d$ features, and $\tilde{x_i}$ is an augmented sample from the data. In order to explain where this augmentation comes from, let's consider what one predicted value would look like if we had already fitted our model.

$$
\huge \hat{y} = w_0 + w_1x_1 + w_2x_2 + ...+w_dx_d
$$

It becomes clear when you consider what this may look like expressed through vectors. In order to maintain this form, we must augment the sample vector $\vec{x} = (x_1, x_2, x_3, ... ,x_d)^T$ so that $\vec{x}^T \vec{w}$ properly results in $\hat{y}$. We do so by prepending a 1 to it such that it becomes $(1, x_1, x_2, x_3, ... ,x_d)^T$. It may be helpful to note that the need for all this augmentation comes from our decision to embed the bias term with the parameter coefficients in $\vec{w}$.

We do this to each sample vector in the data, giving us

$$
\huge \tilde{X} := \begin{bmatrix} \tilde{x_1} \\\ \tilde{x_2} \\\ \vdots \\\ \tilde{x_n} \end{bmatrix} = \begin{bmatrix} 1 & x_{1, 1} & x_{1, 2} & \cdots & x_{1, d} \\\ 1 & x_{2, 1} & x_{2, 2} & \cdots & x_{2, d} \\\ \vdots & \vdots & \vdots & \ddots & \vdots  \\\ 1 & x_{n, 1} & x_{n, 2} & \cdots & x_{n, d} \end{bmatrix}
$$

With this convention, we can represent all of our predicted values with an elegant expression:

$$
\huge \hat{Y} = \tilde{X} \vec{w} \tag{1}
$$

Now that we justified the augmentaion, we can show how to actually find $\vec{w}$. As I mentioned, the idea is the same; We must find where $\nabla L(\vec{w}) = \vec{0}$. It will make the calculations nicer to express $L(\vec{w})$ differently. Note that 

$$
\begin{aligned}
\huge L(\vec{w}) &\huge= \frac{1}{2n} \sum_{i=1}^{n} (y_i - \tilde{x_i}^T\vec{w})^2 \\
&\huge = \frac{1}{2n} \lVert\vec{y} - \tilde{X}\vec{w}\rVert_2^2 \\
&\huge = \frac{1}{2n} (\vec{y} - \tilde{X}\vec{w})^T(\vec{y} - \tilde{X}\vec{w}) \\
&\huge = \frac{1}{2n} (\vec{y}^T \vec{y} - \vec{y}^T \tilde{X}\vec{w} - (\tilde{X}\vec{w})^T \vec{y} + \vec{w}^T\tilde{X}^T\tilde{X}\vec{w}) \\
&\huge = \frac{1}{2n} (\vec{y}^T \vec{y} - 2\vec{w}^T\tilde{X}^T\vec{y} + \vec{w}^T\tilde{X}^T\tilde{X}\vec{w}).
\end{aligned}
$$

To clarify, 

$$
\vec{y} = \begin{bmatrix} y_1 \\\ y_2 \\\ \vdots \\\ y_n \end{bmatrix}
$$ 

is the collection of all true $y_i$ values. Now we can find the gradient of our loss function. 

$$
\begin{aligned}
\huge \nabla L(\vec{w}) &\huge= \nabla_{\vec{w}} (\frac{1}{2n} (\vec{y}^T \vec{y} - 2\vec{w}^T\tilde{X}^T\vec{y} + \vec{w}^T\tilde{X}^T\tilde{X}\vec{w})) \\
&\huge = \frac{1}{2n} (\nabla_{\vec{w}} (\vec{y}^T \vec{y}) - 2\nabla_{\vec{w}} (\vec{w}^T\tilde{X}^T\vec{y}) + \nabla_{\vec{w}} (\vec{w}^T\tilde{X}^T\tilde{X}\vec{w})) \\
&\huge = \frac{1}{2n} (-2\tilde{X}^T\vec{y} + 2\tilde{X}^T\tilde{X}\vec{w}) \\
&\huge = \frac{1}{n} (\tilde{X}^T\tilde{X}\vec{w} - \tilde{X}^T\vec{y}).
\end{aligned}
$$

Like the univariate case, all we have left to do now is solve for when $\nabla L(\vec{w}) = \vec{0}$. We assume $(\tilde{X}^T\tilde{X})$ is invertible. Observe that 

$$
\begin{aligned}
&\huge \nabla L(\vec{w}) = \vec{0} \\
\huge \Rightarrow &\huge \frac{1}{n} (\tilde{X}^T\tilde{X}\vec{w} - \tilde{X}^T\vec{y}) = \vec{0} \\
\huge \Rightarrow &\huge \tilde{X}^T\tilde{X}\vec{w} - \tilde{X}^T\vec{y} = \vec{0} \\
\huge \Rightarrow &\huge \tilde{X}^T\tilde{X}\vec{w} = \tilde{X}^T\vec{y} \\
\huge \Rightarrow &\huge \vec{w} = (\tilde{X}^T\tilde{X})^{-1} \tilde{X}^T\vec{y}. \\
\end{aligned}
$$

We can now use our learned $\vec{w}$ to predict on new data as seen in equation (1).

## Usage

1. Import ```LinearRegression()``` from ```homemade_linear_regressor.py``` and initialize your model.

```python
from homemade_linear_regressor import LinearRegression

lr = LinearRegression()
```

2. Fit your model with the ```fit``` method using your training features and targets. Both should be arrays of the same length and at most 2 dimensional.

```python
# example features
X = [
    [2, 4, 7],
    [10, 1, 25],
    [0, 8, 9]
]

# example targets
y = [3, 10, 4]

lr.fit(X, y)
```

3. Use your model to predict on new data with the ```predict``` method and retrieve a prediction vector. This data should also be an array of at most 2 dimensions.

```python
# example data
data = [
    [25, 1, 3],
    [2, 5, 9],
    [1, 4, 4]
]

y_pred = lr.predict(data)
```