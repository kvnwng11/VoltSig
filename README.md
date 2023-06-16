# VoltSig

In 2022, I worked with [Dr. Qi Feng](https://sites.google.com/site/qifengmath/home) at the University of Michigan to study Volterra signatures. This repository contains our code to solve Volterra signatures.

---

## Motivation

We are interested in studying stochastic Volterra integrals. Let $T > 0$ be a fixed terminal time, $B^0_t \coloneqq t$, and $B = (B^1, \cdots, B^d)$ be a $d$-dimensional Brownian motion. 

A general Volterra type stochastic differential equation (SDE) under Stratonovich integration has the following form, for a $d_1$-dimensional state process $X = (X^1, \cdots, X^{d_1})$ with the initial point given by $x = (x_1, \cdots, x_{d_1}) \in \mathbb{R}^{d_1}$:

$$
    X_t^i = x_i + \sum^d_ {j=0} \int^t_ 0 K_i(t, r) V^i_j(X_r) \circ dB_r^j \quad 1,\cdots,d_1
$$

where $V_j$ are appropriate deterministic functions, and $K_i$ is a deterministic mapping  $\lbrace (t,r) \mid 0 \leq r \leq t \leq T \rbrace \rightarrow [0, \infty)$.

**Example**. Volterra integrals appear often in stochastic volatility models, of which have grown in popularity in recent years. [Comte and Renault](https://onlinelibrary.wiley.com/doi/10.1111/1467-9965.00057) proposed replacing $\sigma$ in the Black-Scholes model with $\sigma_t$, where $\sigma_t$ satisfies the following Stochastic Volterra Integral Equation:

$$
\sigma_t = \sigma_0 + \int_0^t K(t, r)V_0(\sigma_r)dr + \int_0^t K(t, r) V_1(\sigma_r) d\tilde{B}_r
$$

Here, $\tilde{B}$ is a Brownian motion possibly correlated with $B$, $V_0$ and $V_1$ are appropriate deterministic functions, and $K$ is a two time variable deterministic function with a Hurst paramter $H > 0$. 

---

## The Volterra signature

Motivated by stochastic Taylor expansions in his paper [(Feng and Zhang, 2023)](https://arxiv.org/pdf/2110.12853.pdf), Dr. Feng proposed the *step-N Volterra signature* with the following form in the space $\bigotimes\limits^N_{n=0}(\mathbb{R}^{d_1+1})^{\otimes n}$:

$$
\sum^N_ {n=0}\sum_ {\vec{i} \in \mathcal{I}_ n, \vec{j} \in \mathcal{J}_ n, \vec{\kappa} \in \mathcal{S}_ n} (\int_ {\mathbb{T}_ n} \mathcal{K}(\vec{i}, \vec{\kappa}; \vec{t}) \circ dB^{\vec{j}}_ {\vec{t}})(e_{j_1} \otimes \cdots \otimes e_{j_n})
$$

where the followng definitions hold:
* $\lbrace e_j \rbrace_ {j=0,1, \cdots d_1}$ is the canonical basis of $\mathbb{R}^{d_1+1}$
* $\mathcal{I}_ n$ is defined as $\mathcal{I}_ n \coloneqq \lbrace 1, \cdots, d_1 \rbrace^n$ with elements $\vec{i} = (i_1, \cdots, i_n) \in \mathcal{I}_ n$
* $\mathcal{J}_ n$ is defined as $\mathcal{J}_ n \coloneqq \lbrace 0, \cdots, d \rbrace^n$ with elements $\vec{j} = (j_1, \cdots, j_n) \in \mathcal{J}_ n$
* $\mathcal{S}_ n$ is a set of mappings for the indices $\mathcal{S}_ n \coloneqq \lbrace \vec{\kappa} = (\kappa_1, \cdots, \kappa_n) \mid \kappa_\ell \in \lbrace 0, 1, \cdots, \ell-1 \rbrace, \ell = 1, \cdots, n \rbrace$
* $\mathcal{K}(\vec{i}, \vec{\kappa}; \vec{t})$ is defined as $\mathcal{K}(\vec{i}, \vec{\kappa}; \vec{t}) \coloneqq \prod\limits^n_ {\ell = 1}K_ {i_\ell}(t_ {\kappa_\ell}, t_\ell)$. 

The standard signature is a special case when $K_ {i_\ell} = 1$ for all $i_\ell$. In our studies, we were interested in a more general setting with $B$ replaced with a discrete rough path $X$. We wanted to investigate how the Volterra signature summarizes features of the path.

---
## Example code


```python
import numpy as np
import matplotlib.pyplot as plt
import voltsig as vs
```

### One dimension

Below is a randomly generated one dimensional path:


```python
X1 = np.random.rand(10)*100
plt.plot(X1)
plt.show()
```


    
![png](example_files/example_6_0.png)
    


Now, let's set $T=1$ and define our kernel function to be 

$$
k(s,t) = 1
$$

and compute the Volterra signature. 


```python
def kernel(arg1, arg2):
    return 1

v = vs.VoltSig(path=X1, kernel=kernel, T=1)
v.calc(level=3)
s = v.get_sig()
print(s)
```

    Begin calculation...
    Calculating VoltSig element...
    Calculating VoltSig element...
    Calculating VoltSig element...
    [-0.7995695714161881, 0.31965574976731403, 0.31965574976731403, -0.08519567028048414, -0.08519567028048414, -0.08519567028048414, -0.08519567028048414, -0.08519567028048414, -0.08519567028048414]
    

The values of the signature and plotted below:


```python
x = [i for i in range(len(s))]
plt.scatter(x, s)
plt.show()
```


    
![png](example_files/example_10_0.png)
    


Now, while keeping $T=1$, let's set our kernel function to be

$$
k(s,t) = |s-t|
$$

and compute the Volterra signature:


```python
def kernel(arg1, arg2):
    return abs(arg1-arg2)

v = vs.VoltSig(path=X1, kernel=kernel, T=1)
v.calc(level=3)
s = v.get_sig()
print(s)
```

    Begin calculation...
    Calculating VoltSig element...
    Calculating VoltSig element...
    Calculating VoltSig element...
    [-4.536463385183461, 10.289750022555102, -1.22784370389762, -15.559691406670606, 0.9120944075069645, -3.1016032949628927, 7.759576892915574, 4.948580339424591, 3.745879190445755]
    


```python
x = [i for i in range(len(s))]
plt.scatter(x, s)
plt.show()
```


    
![png](example_files/example_13_0.png)
    


As we can see, using a different kernel function changes how the signature summarizes the path. Further research is needed to evaluate this effect.

### Three dimensions

Below is a randomly generated path in three dimensions:


```python
length_of_path = 8

X1 = np.random.rand(length_of_path)*100
X2 = np.random.rand(length_of_path)*100
X3 = np.random.rand(length_of_path)*100
fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111, projection='3d')
ax.plot(X1, X2, X3)
plt.show()
```


    
![png](example_files/example_16_0.png)
    


Again, let's set $T=1$ and our kernel function to be

$$
k(s,t) = |s-t|
$$

and compute the Volterra signature:


```python
def kernel(arg1, arg2):
    return abs(arg1-arg2)

path = [X1, X2, X3]

v = vs.VoltSig(path=path, kernel=kernel, T=1)
v.calc(level=3)
s = v.get_sig()
print(s)
```

    Begin calculation...
    Calculating VoltSig element...
    Calculating VoltSig element...
    Calculating VoltSig element...
    Calculating VoltSig element...
    Calculating VoltSig element...
    Calculating VoltSig element...
    Calculating VoltSig element...
    Calculating VoltSig element...
    Calculating VoltSig element...
    Calculating VoltSig element...
    Calculating VoltSig element...
    Calculating VoltSig element...
    Calculating VoltSig element...
    Calculating VoltSig element...
    Calculating VoltSig element...
    Calculating VoltSig element...
    Calculating VoltSig element...
    Calculating VoltSig element...
    Calculating VoltSig element...
    Calculating VoltSig element...
    Calculating VoltSig element...
    Calculating VoltSig element...
    Calculating VoltSig element...
    Calculating VoltSig element...
    Calculating VoltSig element...
    Calculating VoltSig element...
    Calculating VoltSig element...
    Calculating VoltSig element...
    Calculating VoltSig element...
    Calculating VoltSig element...
    Calculating VoltSig element...
    Calculating VoltSig element...
    Calculating VoltSig element...
    Calculating VoltSig element...
    Calculating VoltSig element...
    Calculating VoltSig element...
    Calculating VoltSig element...
    Calculating VoltSig element...
    Calculating VoltSig element...
    [-0.7531472359875967, -5.2080379255612685, 2.4010314565714013, 0.28361537953788296, -1.6439817686168243, -6.648305209347748, 0.5326859726718223, 0.3422682823027047, 0.7668535857492965, 10.570724577902771, 4.743777782655528, 13.561829517042254, -4.6125500287277745, -21.08067286724603, -0.28955685046743945, -2.15059848733873, -0.7049450895417962, 8.576009980956561, 1.5127293322100162, 2.882476027722694, -1.7172949914593951, -0.07120137972751196, 0.24136999604022077, 1.6396231343999856, -0.6428328053924445, -0.24399294824992682, 0.7554203329673214, -2.9896255818747273, -0.7261269824337223, -0.97422549693368, 0.4787691538664403, -0.009454062975407572, 0.23067063936647808, 0.3344244866740048, -0.05239334409391614, -0.351486473802575, 0.27003567580713744, 0.15991232528793034, -0.029057453901518877, 10.986403856171638, 6.254125835117978, -0.1538331393998645, 5.791318331482334, 5.927210847648054, -0.6166406430355116, -4.6787638422155045, -3.13653926491255, 1.4553565066531953, -4.9291925328812205, 0.15825154290577356, -0.3372967613154741, -16.17468743970445, 0.25909992052303055, -1.592278154789484, 1.7554200272932148, 0.4272234127699935, -0.0959580480192954, -0.9266273841305026, -0.6994646150261434, -0.7951959902274764, -0.1694542695836224, -1.203169316059832, -0.2651856447849467, 9.292903111246826, 3.8170316974443255, 1.976587435067104, 2.007183905450894, 0.4827699283221432, 0.16673964307368302, 3.6494925365905075, -1.1621627454240349, 0.47041415692159466, -1.9202951915733892, -0.9530896025907272, -0.28771828922776477, -9.473857927202644, -4.495023851105625, -4.869057605523099, 3.282019061703401, -1.2880221544865946, 2.9079853072859394, 43.982153355420614, -1.6129630579670806, -0.5090786369433691, -1.0930659898299913, -1.37362066268764, 0.010818431193732877, 5.099238133508421, 5.709026607406511, 11.331279532498439, -6.229489913525107, -1.4469369471259255, -0.6072369884332016, -49.51744392890046, -9.846752533535213, 10.016084895839144, -14.349943432026313, -7.873436417434382, 5.512893997348064, -23.543507488250842, 9.618266841798402, -7.954922845459117, 22.358991486823623, 5.810530372756549, 4.7858017995660544, 92.63192951599751, 9.75526961056625, 6.646167690338028, 1.34601954540046, -3.3428371493792897, -1.76308237482774, 26.9522998089211, 5.8803040197392065, -7.246367693108912, 9.550109387417628, 11.750085761810812, -3.576562325430491, -75.47491524302734, -14.025699075482905, -9.593266097126245, -8.713947753752192, -0.6545098552338291, -4.281514775395545, -20.645828273116223, 3.454674168057692, -3.4055866191259105, 11.305264027805226, 4.526053529545788, 4.445003240621625, 1.273172345294388, -0.011615768419725604, 0.7119974837147781, -2.8963008516501616, -0.5320059184939406, -2.1726875995156596, -9.081005612073822, 2.6228711168190824, 3.4666401931596558, -0.9457577657954536, -0.17164899585849622, -0.10198868945487835, -6.477188160785567, -2.9382048086130097, -5.485465761352795, 2.532988170471075, 0.9872198295604591, -0.014272782268708263, -6.6708957121324595, -9.3453688188054, -5.769362937621322, -5.8216238730971535, -7.3196454477683535, -2.2456179919130665, 15.405365006106825, -0.4286317634234471, 1.2642277839169318, -3.695346328839842, -3.8368052111066695, -2.0024867814994605, -9.323702133716433, -2.748231401324358, -0.6062260253218188, -0.7443226553698259, 1.9479765783552878, 1.3976827206327085, 0.6567667711151938, 3.5688776858456652, 6.151875976379389, 0.2239927403617258, -0.6941313459943027, 2.8069910308954418, 14.95748593493172, 1.2925769674014478, 0.38670481841606996, 2.945759769834869, 1.2082096291954991, 2.0398876208494916, 2.306971871791724, -1.1419485759168408, 0.980015619731089, -3.961346338520768, -1.9510333907850226, -1.8393821428728343]
    

The Volterra signature is plotted below:


```python
x = [i for i in range(len(s))]
plt.scatter(x, s)
plt.show()
```


    
![png](example_files/example_20_0.png)
    

