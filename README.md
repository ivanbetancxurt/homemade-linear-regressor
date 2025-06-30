# Homemade Linear Regressor

This project marks the beginning of a journey towards a goal I have set for myself: **Gain a deep understanding of the theory behind the several foundational machine learning models by building them from scratch.** I've always wanted to intimately understand all the math that goes into these imprssive algorithms and I think this is an exciting way to do it. The rest of this README, as well as those for my other  homemade models, is nothing novel and more just a place for me to document everything I've learned.

## Motivation

The idea of linear regression dates back to more than 2 centuries ago. A cornerstone of statistics, this model is elegant, fast, and easily interpretable, making it a great first choice for exploratory data analysis.

## Mathematical Foundations

### Univariate Case

#### Loss Function

There are several loss functions one may choose to optimize when fitting their model. Here we choose the **sum of the squared residuals.** To breifly explain this loss function, suppose we have some 2-dimensional data i.e. a set of $(x, y)$ coordinates, and any line through that data. For each data point, this function will take the difference between the point's $y$ value and the value of the line at the point's $x$ value. Squaring this difference gives us the *squared residual* for that point. The sum of all the squared residuals is the calculated loss. 

#### Minimizing Loss

Since we are only concerned with the univariate case at the moment, we can define our loss function like this:

$$ 
\huge L(m, b) = \frac{1}{2n} \sum_{i=1}^{n} (y_i - (mx_i + b))^2
$$

where $n$ is the number data points, $y_i$ is the $y$ value of the $i$ th data point (what we may call the "true" value later on), and $m, b$ are the slope and intercept terms in the classic definition of a line ($y = mx + b$) respectively. 

*Note: The added factor of* $\frac{1}{2n}$ *is there simply to make differentiation down the line neater. Though it may change what the actual calculated loss is at a given point, it does* not *change where minimum of the function lays, which is what we are interested in.*

Semantically, this function asks, "how well does my line $mx + b$ model the data?" Of course, a lower value means that line models the data better. Hence, for the best model, we must minimize the value of $L(m, b)$.

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

All that's left to do is set these derivative function equal to 0 and solve the system of equations. Then we will have found the values for $m$ and $b$ that satisfy $\nabla L(m, b) = (0, 0)$. Note that

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

We can use matrices to solve the system. Let $A = \begin{bmatrix} \sum x_i^2 & \sum x_i \\ \sum x_i & n \end{bmatrix}, \vec{c} = \begin{pmatrix} \sum x_iy_i \\ \sum y_i \end{pmatrix}$. We assume $A$ is invertible. Observe that

$$
\begin{aligned}
&\huge A \begin{bmatrix} m \\\ b \end{bmatrix} = \vec{c} \\
\huge \Rightarrow &\huge A^{-1} (A \begin{bmatrix} m \\\ b \end{bmatrix}) = A^{-1} \vec{c} \\
\huge \Rightarrow &\huge \begin{bmatrix} m \\\ b \end{bmatrix} = A^{-1} \vec{c} \\
\huge \Rightarrow &\huge \begin{bmatrix} m \\\ b \end{bmatrix} = \frac{1}{n \sum x_i^2 - (\sum x_i)^2} \begin{bmatrix} n & -\sum x_i \\\ -\sum x_i & \sum x_i^2 \end{bmatrix} \begin{bmatrix} \sum x_iy_i \\\ \sum y_i \end{bmatrix} \\
\huge \Rightarrow &\huge m = \frac{\sum x_iy_i - n\bar{x}\bar{y}}{\sum x_i^2 - n\bar{x}^2}, b = \bar{y} - m\bar{x}
\end{aligned}
$$

where $\bar{x} = \frac{1}{n} \sum_{i} x_i$ and $\bar{y} = \frac{1}{n} \sum_{i} y_i$ are the means of all ${x_i}$ and ${y_i}$ respectively. 