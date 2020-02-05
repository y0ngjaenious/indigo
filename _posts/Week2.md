## Machine Learning Study

**Week2, 2020.01.28**



## 3. Naive bayes Classifier

### 3.1 What is the optimal classifier?

Recall that the goal of machine learning is to "**find a function** $f$ **whose error is within PAC bounds**".

**Definition of Bayes Classifier :** 
$$
f^*=\text{argmin}_fP(f(X)\neq Y)
$$
The aim of Bayes Classifier is function approximation by error minimization. Note that optimal classifier SHOULD have some error, $R(f^*)>0$. We may make decision with a decision boundary. The area under the probability curve of classes which was not chosen by decision is Bayes Risk. The formula above is a form of function-approximation, and we now use a form of decision. That means,
$$
f^*(x)=\text{argmax}_{Y=y}P(Y=y\mid X=x)
$$
And by using Bayes' theorem, we can obtain the formula as follows:
$$
f^*(x) = \text{argmax}_{Y=y}P(Y=y\mid X=x)= \text{argmax}_{Y=y}P(X=x\mid Y=y)P(Y=y)
$$
However, it is impossible to know what $P(X=x\mid Y=y)$ is. So, we now apply some relaxed assumption, that is,

> All independent variables $x_1,x_2,\cdot\cdot\cdot,x_d$ are conditionally independent with given $y$

### 3.2 Naive Bayes Classifier

#### Quick Note for conditionally independent vs marginal independence

- $X_1$ and $X_2$ are ***marginally independent*** iff $P(X_1) = P(X_1\mid X_2)$
  - That means, knowledge of $X_2$ doesn’t affect your belief in $X_1$.
- $X_1$ and $X_2$ are ***conditionally independent*** with given $Y$ iff $P(X_1\mid Y) = P(X_1\mid Y, X2)$
  - That means, knowledge of $X_2$ doesn’t affect your belief in $X_1$, **given ** $Y$.

By the assumption, we can approximate the model like following:
$$
f^*(x)=\text{argmax}_{Y=y}P(X=x\mid Y=y)P(Y=y) \approx \text{argmax}_{Y=y}P(Y=y)\prod_{1\le i\le d}P(X_i=x_i\mid Y=y)
$$
What is the **purpose of this assumption**? That is, we can reduce the number of parameter. Before the assumption, our parameter space was $O(k\cdot2^d)$ where $k$ is the number of classes and $b$ is the number of independent variables(the dimension of feature space). After the assumption, our parameter space reduced to $O(k\cdot d)$. Then, we now want to formulate Naive Bayes Classifier.

---

Given : 

- Class Prior $P(Y)$
- $d$ conditionally independent variables $X$ given the class $Y$
- For each $X_i$, we have the likelihood of $P(X_i\mid Y)$

The **Naive Bayes Classifier Function** is,
$$
f_{NB}=\text{argmax}_{Y=y}P(Y=y)\prod_{1\le i\le d}P(X_i=x_i\mid Y=y)
$$

---

If the following two conditions are met, the Naive Bayes classifier is the best classifier:

- conditional independent assumption on X hold
- Exact prior

### 3.3 Problem of Naive Bayes Classifier

- correlated $X$
- For a sparse Data, there will be some point that was not be observed. MLE will set the probability of that point as zero. Therefore, the performance of MLE parameter estimation can be low.
- The alternative is MAP, but it also works not that good with incorrect prior.



Bag Of Words?



## 4. Logistic Regression

Recall the plot of the decision boundary and Bayes Risk. For the reason of range(linear function will violate the probability axiom) and risk optimization, we want to use curve rather than linear functions. The curve is required the following properties:

- Bounded [0,1]
- Differentiable
- Real function
- Defined for all real inputs
- With positive derivative

Sigmoid function satsatisfies all these properties, and **Logistic function** is a special case of sigmoid function.

### 4.1 Logistic function

---

$$
f(x)={1\over1+e^{-x}}
$$

---

The inverse of the logistic function is a logit function. Start from logit,
$$
f(x)=\log({x\over1-x})\\
\quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad x = \log({p\over 1-p}) \quad(\text{inverse of X and B})\\
\quad\quad \quad \quad \quad \; \; ax+b = \log({p\over1-p}) \quad (\text{linear shift})\\
\quad\quad\quad\quad \quad \quad \quad\quad\quad\;\;\;\quad \quad \quad X\theta = \log({p\over1-p})\quad \text{(turn to the form of MLR)}
$$
The purpose of this transformation is following:

- Using (Multivariate) Linear Regression for probability($\text{i.e.} \;\; X\theta=P(Y\mid X)$) will violate the probability axiom.
- So, we now use the logistic function to approximate $P(Y\mid X)$, that is $X\theta=\log{P(Y\mid X)\over1-P(Y\mid X)}$

Given the Bernoulli experiment,
$$
P(y\mid x) = \mu(x)^y(1-\mu(x))^{1-y}
$$
and where $\mu(x)$ is the probability of $y=1$ given $x$. Then, set $\mu(x) = {1\over 1+e^{-\theta^Tx}}={e^{X\theta}\over 1+e^{X\theta}}$.

Finally, the goal becomes finding the parameter $\theta$.

### 4.2 Finding the parameter $\theta$

We will use **MCLE(Maximum Conditional Likelihood Estimation)**. That is,
$$
\hat{\theta} = \text{argmax}_\theta P(D\mid \theta) = \text{argmax}_\theta\prod_{1\le i\le N}\log(P(Y_i\mid X_i;\theta))\\
\quad\quad\quad\quad\quad\quad\quad\quad\quad\;=\text{argmax}_\theta\log(\prod_{1\le i\le N}P(Y_i\mid X_i;\theta))\\
\quad\quad\quad\quad\quad\quad\quad\quad\quad\;=\text{argmax}_\theta\sum_{1\le i \le N}\log(P(Y_i\mid X_i;\theta))
$$
From the previous formula, $P(Y_i\mid X_i;\theta)=\mu(X_i)^{Y_i}(1-\mu(X_i)^{Y_i})$, therefore
$$
\log(P(Y_i\mid X_i;\theta))\;\; = Y_i\log(\mu(X_i)^{Y_i})+(1-Y_i)\log(1-\mu(X_i))\\
\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\;\;\;\quad\;=Y_i\lbrace \log(\mu(X_i))-\log(1=\mu(X_i))\rbrace + \log(1-\mu(X_i))\\
\;\;\quad\quad\quad\quad\quad\;=Y_i\log({\mu(X_i)\over1-\mu(X_i)})+ \log(1-\mu(X_i))\\
\;\;=Y_iX_i\theta+ \log(1-\mu(X_i))\\
=Y_iX_i\theta-\log(1+e^{X_i\theta})
$$
Then,
$$
\hat{\theta} = \text{argmax}_\theta\sum_{1\le i \le N}\log(P(Y_i\mid X_i;\theta)) = \text{argmax}_\theta\sum_{1\le i \le N}\lbrace Y_iX_i\theta-\log(1+e^{X_i\theta})\rbrace
$$
We now use partial derivative to find $\theta$
$$
{\partial\over\partial\theta_j}\lbrace \sum_{1\le i \le N}(Y_iX_i\theta-\log(1+e^{X_i\theta}))\rbrace = \lbrace \sum_{1\le i \le N} Y_iX_{i,j}\rbrace+\lbrace\sum_{1\le i \le N} -{e^{X_i\theta}X_{i,j}\over1+e^{X_i\theta}}\rbrace\\
\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\;\;=\sum_{1\le i \le N}X_{i,j}(Y_i-{e^{X_i\theta}\over1+e^{X_i\theta}})\\
\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad=\sum_{1\le i \le N}X_{i,j}(Y_i-P(Y_i=1\mid X_i;\theta))
$$
We cannot derive further. There is no closed form solution. Therefore, we just use open form solution and do approximation.

### 4.3 Gradient Method

Now, we will use **gradient method** to approximate the parameter $\theta$ . Before talking about gradient descent/ascent algorithm, let's do quick recap about Taylor Expansion(Taylor Series).

#### Quick Note for Taylor expansion

A **Taylor series** is a representation of a function as an infinite sum of terms that are calculated from the values of the function's derivatives at a single (fixed) point.

That is, we can represent a function $f(x)$ as follows:
$$
f(x) = f(a)+{f'(a)\over1!}(x-a)+{f''(a)\over2!}(x-a)^2+\cdot\cdot\cdot = \sum_{n=0}^{\infty}{f^{(n)}(a)\over n!}(x-a)^n
$$
The function MUST be infinitely differentiable at a real or complex number of $a$ for taylor expansion.

In fact, Taylor expansion is a **process** of generating the Taylor series.

#### Gradient Descent/Ascent

Gradient descent/ascent algorithm is,

---

**Algorithm**

---

- Given a differentiable function $f(x)$ and an initial parameter $x_1$, and $h$(learning rate)
- Iteratively moving the parameter to the lower/higher value of $f(x)$
  - By taking the direction of the negative/positive gradient of $f(x)$

---

The idea of the algorithm is, we can represent $f(x)$ as,
$$
f(x) = f(a)+{f'(a)\over1!}(x-a)+{f''(a)\over2!}(x-a)^2+\cdot\cdot\cdot = f(a)+{f'(a)\over1!}(x-a)+O(\mid\mid x-a\mid\mid^2)
$$

##### Quick Note about Big-O Notation

**Definition**

---

For $f, g : \N\rightarrow\R^+$, if there exist constants $k$ and $C$ such that $f(n)\le C\cdot g(n)$ for all $n>k$, we say $f(n)=O(g(n))$

*c.f.*

For $f,g : \N\rightarrow \R^+$, if there exist constants $k$ and $C$ such that $f(n)\ge C\cdot g(n)$ for all $n>k$ , we say $f(n) = \Omega(g(n))$

If $f(n)=O(g(n))$ and $f(n)=\Omega(g(n))$, we say $f(n)=\Theta(g(n))$

---

Let me give you some example : 

>  $2n = O(n^2)$,	$2n\neq\Omega(n^2)$,	$2n\neq\Theta(n^2)$

>  $2n = O(n)$,	$ 2n=\Omega(n)$,	$2n=\Theta(n)$

>  $2n^2 = O(n^2)$,	$2n^2=\Omega(n^2)$,	$2n^2=\Theta(n^2)$

> $2n^2 \neq O(n)$,	$2n^2=\Omega(n)$,	$2n^2\neq\Theta(n)$

Plus, the nice thing of Big-O notation is that it can be used to the term that "dominates" the speed of growth. It is called "asymptotic notation". So, we now treat $2n$ and $3n$ "equally" because $2n=O(n),\;\;3n=O(n)$



What we want to know is the direction of move! Set $a=x_1$ and $x=x_1+h\mathbf{u}$, where $\mathbf{u}$ is the unit **direction** vector for the partial derivative. Then, we can substitue this into the formula above.
$$
f(x)=f(a)+{f'(a)\over1!}(x-a)+O(\mid\mid x-a\mid\mid^2) \Rightarrow f(x_1+h\mathbf{u})=f(x_1)+hf'(x_1)\mathbf{u}+h^2O(1)
$$
Let's assume that $h$ is a small value, then we can erase the last term of the equation. That is,
$$
f(x_1+h\mathbf{u})-f(x_1)\approx hf'\mathbf(x_1){u}
$$
Then, we want to find the direction of move, $\mathbf{u}$. Our desired direction is to minimize the difference between $f(x_1+h\mathbf{u})$ and $f(x_1)$. Therefore,
$$
\mathbf{u}^* =\text{argmin}_{\mathbf{u}}\lbrace f(x_1+h\mathbf{u})-f(x_1)\rbrace = \text{argmin}_\mathbf{u}hf'(x_1)\mathbf{u}=-{f'(x_1)\over \mid f'(x_1)\mid}
$$
The reason $\mathbf{u}^*$ becomes $-{f'(x_1)\over\mid f'(x_1)\mid}$ is as follows:

- h is a constant
- The dot product of two vectors is minimal when the directions of the two vectors are completely opposite.
- since $\mathbf{u}$ is an unit vector, divide by its norm to normalize

So **Gradient descent** algorithm update $x_{t+1}\leftarrow x_t+h\mathbf{u}=x_t-h{f'(x_1)\over \mid f'(x_1)\mid}$

For approximation of parameter $\theta$ in Logistic Regression which is find $\hat{\theta}$,
$$
\hat{\theta} = \text{argmax}_\theta\sum_{1\le i \le N}\log(P(Y_i\mid X_i;\theta))
$$
we need to use **Gradient ascent** algorithm. That is just, taking $f(\theta)$ as $f(\theta)=\sum_{1\le i \le N}\log(P(Y_i\mid X_i;\theta))$
$$
x_{t+1}\leftarrow x_t+h\mathbf{u}=x_t+h{f'(x_1)\over \mid f'(x_1)\mid}
$$
which means taking the direction of the positive gradient of $f(\theta_t)$

### 4.4 Find $\theta$ with Gradient Ascent Algorithm

- Set $f(\theta)=\sum_{1\le i \le N}\log(P(Y_i\mid X_i;\theta))$
- Recall that partial derivative is ${\partial f(\theta)\over\partial\theta_j} = {\partial\over\partial\theta_j}\lbrace\sum_{1\le i \le N}\log(P(Y_i\mid X_i;\theta))\rbrace =\sum_{1\le i \le N}X_{i,j}(Y_i-P(Y_i=1\mid X_i;\theta))$

Then the algorithm of apporoximation is as follows:

---

- $\theta_j^0$(initial parameter) is arbitrary chosen.
- Iteratively moving $\theta_t$ to the higher value of $f(\theta_t)$ by taking the direction of the positive gradient of $f(\theta_t)$ as follows:

$$
\theta_j^{t+1}\leftarrow \theta_j^t + h{\partial f(\theta^t)\over\partial\theta_j^t} = \theta_j^t+h\lbrace\sum_{1\le i \le N}X_{i,j}(Y_i-P(Y_i=1\mid X_i;\theta^t))\rbrace \\
\quad\quad\;\;\;\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad= \theta_j^t+{h\over C}\lbrace \sum_{1\le i \le N}X_{i,j} (
Y_i-{e^{X_i\theta^t}\over1+e^{X_i\theta^t}})\rbrace\quad\text{(C = normalizing factor for unit vector.)}
$$

- Repeat the update until converges



#### Find $\theta$ of Linear Regression with Gradient Descent!

In lieu of using least square solution($\hat{\theta}=(X^TX)^{-1}X^TY$), we can use gradient descent algorithm to find the solution.
$$
\hat{\theta} = \text{argmin}_\theta(f-\hat{f})^2 = \text{argmin}_\theta(Y-X\theta)^2 = \text{argmin}_\theta\sum_{1\le i\le N}(Y^i-\sum_{i\le j\le d}X^i_j\theta_j)^2
$$
Set $f(\theta) =\sum_{1\le i\le N}(Y^i-\sum_{i\le j\le d}X^i_j-\theta_j)^2$, then
$$
{\partial f(\theta)\over\partial\theta_k} = {\partial \over\partial\theta_j}\lbrace\sum_{1\le i\le N}(Y^i-\sum_{i\le j\le d}X^i_j\theta_j)^2\rbrace = -\sum_{1\le i \le N}2(Y^i-\sum_{i\le j\le d}X^i_j\theta_j)X_k^i
$$
Therefore, we can iteratively move $\theta_k^t$ to the lower value of $f(\theta^t)$ by taking the direction of the negetive gradient of $f(\theta^t)$ as follows:
$$
\theta_k^{t+1}\leftarrow\theta^t_k-h{\partial f(\theta^t)\over\partial\theta_k^t} = \theta^t_k+h\sum_{1\le i \le N}2(Y^i-\sum_{i\le j\le d}X^i_j\theta_j)X_k^i
$$

### 4.5 The difference between the naive Bayes and the Logistic Regression

#### Naive Bayse to Logistic Regression

Since we cover the naive Bayes with categorical independent variables, we now expand to the continuous feature space for comparing to the logistic regression. For this, we assume that independent variables follow the Gaussian Distribution with the parameter $\mu, \sigma^2$. That is,
$$
P(X_i\mid Y,\mu,\sigma^2) = {1\over\sigma\sqrt{2\pi}}e^{-{(X_i-\mu)^2\over2\sigma^2}}
$$
Let $\pi_1$ be the prior of $Y=y$, ($\text{i.e.}\;\;P(Y=y)=\pi_1$), then the **Gaussian Naive Bayes** model is
$$
f_{NB}(x)=\text{argmax}_{Y}P(Y)\prod_{1\le i \le d}P(X_i\mid Y)=\pi_k\prod_{1\le i\le d}{1\over\sigma_k^iC}\exp{(-{1\over2}({(X_i-\mu_k^i)^2\over\sigma_k^2}))}
$$
where $k$ means class and $C=\sqrt{2\pi}$.

Recall that the naive Bayes assumption is, $P(X\mid Y=y) = \prod_{1\le i \le d}P(X_i\mid Y=y).$ With these two equation, we can convert the posterior as follows :
$$
P(Y=y\mid X)={P(X\mid Y=y)P(Y=y)\over P(X)} = {P(X\mid Y=y)P(Y=y)\over P(X\mid Y=y)P(Y=y)+P(X\mid Y=n)P(Y=n)}\\
\quad\;\;= {P(Y=y)\prod_{1\le i \le d}P(X_i\mid Y=y)\over P(Y=y)\prod_{1\le i \le d}P(X_i\mid Y=y)+P(Y=n)\prod_{1\le i \le d}P(X_i\mid Y=y)}\\
\quad\quad\;\;\;= {\pi_1\prod_{1\le i\le d}{1\over\sigma_1^iC}\exp{(-{1\over2}({(X_i-\mu_1^i)^2\over\sigma_1^2}))}\over \pi_1\prod_{1\le i\le d}{1\over\sigma_1^iC}\exp{(-{1\over2}({(X_i-\mu_1^i)^2\over\sigma_1^2}))}+\pi_2\prod_{1\le i\le d}{1\over\sigma_2^iC}\exp{(-{1\over2}({(X_i-\mu_2^i)^2\over\sigma_2^2}))}}\\
={1\over1+{\pi_2\prod_{1\le i\le d}{1\over\sigma_2^iC}\exp{(-{1\over2}({(X_i-\mu_2^i)^2\over\sigma_2^2}))}\over \pi_1\prod_{1\le i\le d}{1\over\sigma_1^iC}\exp{(-{1\over2}({(X_i-\mu_1^i)^2\over\sigma_1^2}))}}}
$$
and we now add one more assumption that two classes have the same variance $\sigma_1^i=\sigma_2^i$, Then,
$$
P(Y=y\mid X) = {1\over1+{\pi_2\prod_{1\le i\le d}{1\over\sigma_2^iC}\exp{(-{1\over2}({(X_i-\mu_2^i)^2\over\sigma_2^2}))}\over \pi_1\prod_{1\le i\le d}{1\over\sigma_1^iC}\exp{(-{1\over2}({(X_i-\mu_1^i)^2\over\sigma_1^2}))}}} ={1\over1+{\pi_2\prod_{1\le i\le d}\exp{(-{1\over2}({(X_i-\mu_2^i)^2\over\sigma_2^2}))}\over \pi_1\prod_{1\le i\le d}\exp{(-{1\over2}({(X_i-\mu_1^i)^2\over\sigma_1^2}))}}}\\
\quad\quad\quad\quad\quad\quad\;\; = {1\over1+{\pi_2\exp{\lbrace-\sum_{1\le i\le d}{1\over2}({(X_i-\mu_2^i)^2\over\sigma_2^2})\rbrace}\over {\pi_1\exp{\lbrace-\sum_{1\le i\le d}{1\over2}({(X_i-\mu_1^i)^2\over\sigma_1^2})\rbrace}}}} = {1\over1+{\exp{\lbrace-\sum_{1\le i\le d}{1\over2}({(X_i-\mu_2^i)^2\over\sigma_2^2})+\log\pi_2\rbrace}\over {\exp{\lbrace-\sum_{1\le i\le d}{1\over2}({(X_i-\mu_1^i)^2\over\sigma_1^2})+\log\pi_1\rbrace}}}}
$$
Finally, we can obtaion
$$
P(Y=y\mid X) = {1\over 1+\exp\lbrace{-\sum_{1\le i\le d}{1\over2}({(X_i-\mu_2^i)^2\over\sigma_2^2})+\log\pi_2+\sum_{1\le i\le d}{1\over2}({(X_i-\mu_1^i)^2\over\sigma_1^2})-\log\pi_1}\rbrace}\\
= {1\over 1+\exp\lbrace{-{1\over2(\sigma_1^i)^2}\sum_{1\le i\le d}\lbrace{2(\mu_2^i-\mu_1^i)X
_i+(\mu_2^i)^2-(\mu_1^i)^2}\rbrace+\log\pi_2-\log\pi_1}\rbrace}
$$
which is the logistic functioin!

#### Naive Bayes versus Logistic Regression

##### Naive Bayes Classifier

Finally, Gaussian Naive Bayes becomes,
$$
P(Y\mid X) = {1\over 1+\exp\lbrace{-{1\over2(\sigma_1^i)^2}\sum_{1\le i\le d}\lbrace{2(\mu_2^i-\mu_1^i)X
_i+(\mu_2^i)^2-(\mu_1^i)^2}\rbrace+\log\pi_2-\log\pi_1}\rbrace}
$$
This equation can be hold with following assumption:

- Naive Bayes Assumption
- Same variance between two classes
- $P(X\mid Y)$ follows Gaussian distribution
- $P(Y)$ follows Bernoulli distribution

And the number of parameters to estimate is $2kd+1$, where $k=\text{number of classes}, d=\text{number of feature variables}$

##### Logistic Regression

$$
P(Y\mid X) = {1\over 1+e^{-\theta^Tx}}
$$

The number of parameters to estimate is $d+1$

It seems that logistic regression is more efficient, but we need to examine the trade-off between two models(in fact, between all possible classifier!)

####  Generative - Discriminative Pair

##### Generative Model

The characteristics of generative model is modeling the joint probability. That is, generative model expand the posterior to following form:
$$
P(Y\mid X)={P(X,Y)\over P(X)} = {P(X\mid Y)P(Y)\over P(X)}
$$
then, model $P(X\mid Y), P(Y)$ respectively(i.e. estimate the parameter of $P(X\mid Y), P(Y)$)

Ex) Naive Bayes Classifier

##### Discriminative Model

The characteristics of discriminative model is modeling the conditional probability. That is, discriminative model directly estimate the parameter of $P(Y\mid X)$

Ex) Logistic Regression



### Reference

- IE661, KAIST, Ilchul Moon
- Pattern recognition and machine learning, Christopher M. Bishop
- CS2103, Yonsei University, Hyung-Chan An