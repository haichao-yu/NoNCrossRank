# NoNCrossRank

This is an implementation of NoNCrossRank algorithm with Python.

The goal of this algorithm is to solve:<br/>
**Given:** (1) an Network of Networks (NoN) R = <G, A, &theta;>, and (2) the query vectors e<sub>i</sub> (i = 1, ..., g);</br>
**Find:** ranking vectors r<sub>i</sub> for the nodes in the domain-specific networks A<sub>i</sub> (i = 1, ..., g).



## Functions

* **__init__.py:** program entry;
* **CR_CrossValidation.py:** cross rank cross validation;
* **CR_Precomputation.py:** precompute the Anorm and Ynorm;
* **CR.py:** ranking process;
* **J_CR.py:** compute objective fuction;
* **AUCEvaluation.py** AUC evaluation process;
* **AUCValue.py:** compute AUC value;



## Input/Output Format

### - Input

G: the adjacency matrix of main network<br/>
A: domain specific networks A = (A<sub>1</sub>, ..., A<sub>g</sub>)<br/>
&theta;: the one-to-one mapping function (mapping main node to domain-specific network)<br/>
e<sub>i</sub>: the query vector for A<sub>i</sub> (i = 1, ..., g)

### - Output
r<sub>i</sub>: the ranking vector for A<sub>i</sub> (i = 1, ..., g)<br/>
AUCValue: the AUC value is used for evaluation



## Reference
Ni, J., Tong, H., Fan, W., & Zhang, X. (2014, August). **Inside the atoms: ranking on a network of networks**. In Proceedings of the 20th *ACM SIGKDD* international conference on Knowledge discovery and data mining (pp. 1356-1365). ACM.