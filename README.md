# Relational Neural Gas

Copyright (C) 2015-2017
Benjamin Paaßen
AG Machine Learning
Centre of Excellence Cognitive Interaction Technology (CITEC)
University of Bielefeld

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

## Introduction

_Relational neural gas_ is a clustering algorithm for distance data, meaning that you put in a
matrix of pairwise distances D as well as a number of clusters K and you receive a distribution of
your data points into K distinct clusters. It has first been proposed by
[Hammer and Hasenfuss (2007)][1] and is an extension of the _neural gas_ algorithm by
[Martinetz and Schulten (1991)][2]. The basic idea of neural gas is to represent each cluster k in
terms of a prototype w<sub>k</sub> which is responsible for all data points for which this
prototype is the closest prototype. This prototype-based representation has two main advantages
compared to other clustering methods:

1. It permits to extend the clustering to new data points by assigning new data points to their
closest prototype (out-of-sample extension).
2. It permits to inspect the data set in terms of representative data points (i.e. the prototypes).

## Installation

This implementation is written for Java 7 and has no external dependencies. It is also fully
compatible with Matlab as it only interfaces with primitive data types. You can access this package
by either downloading the finished .jar file <!--TODO: Download link-->
or declaring a maven dependency to

<pre>
&lt;dependency&gt;
	&lt;groupId&gt;de.cit-ec.ml&lt;/groupId&gt;
	&lt;artifactId&gt;rng&lt;/artifactId&gt;
	&lt;version&gt;1.0.0&lt;/version&gt;
	&lt;scope&gt;compile&lt;/scope&gt;
&lt;/dependency&gt;
</pre>

If you want to compile the package from source you can download the source code via `git pull` <!-- TODO GIT LINK--> and use the command `mvn package` to compile a .jar distribution.

You can download the javadoc either via maven or directly as a .zip file <!-- TODO Link -->,
or compile it yourself by downloading the source code via `git pull` and using the command
`mvn generate-sources javadoc:javadoc`.

If you want to use the package from MATLAB, please download the .jar distribution and add the
line

<pre>javaaddpath rng-1.0.0.jar;</pre>

to your MATLAB script.

## Quickstart

You can obtain a clustering by using the `RelationalNeuralGas.train()` method. In particular,
if you have a data set given in term of a distance matrix D and a number of desired clusters K,
you can call:

<pre>
final RNGModel model = RelationalNeuralGas.train(D, K); // Java
model = de.citec.ml.rng.RelationalNeuralGas.train(D, K); % MATLAB
</pre>

This results in a `RNGModel` object which can be used for further queries. In particular:

<pre>final int[] assignments = RelationalNeuralGas.getAssignments(model); // Java
assignments = de.citec.ml.rng.RelationalNeuralGas.getAssignments(model) %MATLAB</pre>

returns the cluster assignments of all data points. Further:

<pre>final int[] assignments = RelationalNeuralGas.classify(D2, model); // Java
assignments = de.citec.ml.rng.RelationalNeuralGas.classify(D2, model) %MATLAB</pre>

assigns new data points to clusters given the distances D2 from the new data points to the training
data points. Finally:

<pre>final int[] exemplars = RelationalNeuralGas.getExemplars(model); // Java
exemplars = de.citec.ml.rng.RelationalNeuralGas.getExemplars(model); % MATLAB</pre>

returns the most representative data point for each cluster, i.e. the data point which is closest
to the prototype for the respective cluster.

In general, it is recommended to use the functions provided by the RelationalNeuralGas class.
It is also possible to directly interact with other classes in this package, but this requires more
detailed knowledge on the subject. Please consult the following Background section and the javadoc
for that purpose.

## Background

This is a short introduction regarding the background of relational neural gas. For a more
comprehensive explanation, I recommend to consult [Hammer and Hasenfuss (2010)][3].

### Neural Gas

Assume that we are given a set of data points x<sub>1</sub>, ..., x<sub>m</sub> as well as a number
of clusters K. Neural gas attempts to find prototypes w<sub>1</sub>, ..., w<sub>K</sub> such that
the quantization error

<center>E = &Sigma;<sub>i=1,...,m</sub> &Sigma;<sub>k=1,...,K</sub> h(k|i) &middot; d(x<sub>i</sub>, w<sub>k</sub>)²</center>

is minimized, where h(k|i) is 1 if prototype k is the closest prototype to data point i and 0
otherwise and d(x<sub>i</sub>, w<sub>k</sub>) is the _distance_ between data point i and prototype k.

The neural gas approach to optimizing this error is to _soften_ the assignment variable h and
anneal it over time to become crisp. In particular, we re-define h(k|i) as a Boltzmann
distribution over the prototype _ranks_, that is, data point i is _mostly_ assigned to its closest
prototype, less to its second-closest prototype, even less to its third-closest prototype and so
on. In formulas, we have:

<center>
h(k|i) := exp(- r(k|i) / &lambda;<sub>t</sub>)

r(k|i) := |{ l | d(x<sub>i</sub>, w<sub>l</sub>) &lt; d(x<sub>i</sub>, w<sub>k</sub>)}|
</center>

where r(k|i) is the rank of prototype k with respect to data point i (starting at 0) which can also
be expressed as the number of prototypes which are _closer_ to data point i than prototype k.
Further, &lambda;<sub>t</sub> regulates the _crispness_ of the assignment. For &lambda; close to
zero, h is 1 for the closest prototype and 0 otherwise. For larger &lambda;, distant prototypes
are considered as well. Neural gas starts at &lambda;<sub>0</sub> = K / 2 and exponentially
decreases &lambda; down to &lambda;<sub>T</sub> = 0.01, which essentially yields a crisp assignment.

Neural gas then optimizes the (softened) quantization error in each iteration by setting:

<center>w<sub>k</sub> = &Sigma;<sub>i=1,...,m</sub> h(k|i) &middot; x<sub>i</sub> /
	&Sigma;<sub>j=1,...,m</sub> h(k|j)</center>

In other words, the prototype becomes the weighted mean of all data points assigned to it. Notice
the striking similarity to K-means clustering. Indeed, neural gas can be seen as a proper
generalization of K-means because it reduces to K-means for &lambda; = 0. Further notice that neural
gas is essentially an expectation maximization scheme where the expectation step consists of
re-computing h(k|i) and the maximization step consists of re-computing the prototype positions.

### Relational Neural Gas

To make neural gas 'relational' we apply two tricks: We represent our prototypes and
the distances to our prototypes _indirectly_.

First, we assume that the prototypes w<sub>k</sub> can be expressed as _convex combinations_ of
data points, that is, all prototypes can be expressed in the form:

<center>w<sub>k</sub> = &Sigma;<sub>i=1,...,m</sub> &alpha;<sub>k, i</sub> &middot; x<sub>i</sub></center>

such that all &alpha;<sub>k, i</sub> are non-negative and the coefficients &alpha;<sub>k, i</sub>
for any k sum up to 1.

It turns out that this is no limitation to neural gas, because the neural gas maximization
step already ensures that we obtain prototypes as convex combinations, where the convex
coefficients &alpha;<sub>k, i</sub> are given as:

<center>&alpha;<sub>k, i</sub> = h(k|i) / &Sigma;<sub>j=1,...,m</sub> h(k|j)</center>

Our second trick is to compute the distances between prototypes and data points indirectly. This
can be done via the following formula (the proof as well as the theory behind it can be found in
[Hammer and Hasenfuss (2010)][3]):

<center>d(x<sub>i</sub>, w<sub>k</sub>)² = <strong>&alpha;<sub>k</sub></strong> &middot; <strong>D</strong>(:, i)²
- 0.5 &middot; <strong>&alpha;<sub>k</sub></strong> &middot; <strong>D</strong>² &middot; <strong>&alpha;<sub>k</sub></strong><sup>T</sup></strong></center>

where <strong>&alpha;<sub>k</sub></strong> is the vector of convex coefficients for prototype k,
D(:, i)² is the entry-wise squared i-th column of the distance matrix D and D² is the entry-wise
squared version of D.

Using these two tricks, we can represent prototypes purely in terms of their convex coefficients
and we can infer data-to-prototype distances purely based on these coefficients as well as the
pairwise data-to-data distances.

## License

This documentation is licensed under the terms of the [creative commons attribution-shareAlike 4.0 international (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/) license. The code
contained alongside this documentation is licensed unter the
[GNU General Public License Version 3](https://www.gnu.org/licenses/gpl-3.0.en.html) license.
A copy of this license is contained in the `gpl-3.0.md` file alongside this README.

## Literature

* Hammer, B. and Hasenfuss, A. (2007). _Relational Neural Gas_. In: Hertzberg, J., Beetz, M., and Englert, R. (eds.). Proceedings of the 30th Annual German Conference on AI (KI 2007). Osnabrück, Germany. pp. 190-204. doi: [10.1007/978-3-540-74565-5_16][1]
* Martinetz, T., and Schulten, K. (1991). _A 'Neural-Gas' Network Learns Topologies_. In: Kohonen, T., Mäkisara, K., Simula, O., and Kangas, J. (eds.). Artificial Neural Networks. Elsevier. pp. 397-402. [Link][2]
* Hammer, B., and Hasenfuss, A. (2010). _Topographic Mapping of Large Dissimilarity Data Sets_. Neural Computation, 22(9). pp. 2229-2284. doi: [10.1162/NECO_a_00012](https://doi.org/10.1162/NECO_a_00012)

[1]: https://doi.org/10.1007/978-3-540-74565-5_16 "Hammer, B. and Hasenfuss, A. (2007). Relational Neural Gas. In: Hertzberg, J., Beetz, M., and Englert, R. (eds.). Proceedings of the 30th Annual German Conference on AI (KI 2007). Osnabrück, Germany. pp. 190-204."
[2]: https://www.ks.uiuc.edu/Publications/Papers/PDF/MART91B/MART91B.pdf "Martinetz, T., and Schulten, K. (1991). A 'Neural-Gas' Network Learns Topologies. In: Kohonen, T., Mäkisara, K., Simula, O., and Kangas, J. (eds.). Artificial Neural Networks. Elsevier. pp. 397-402."
[3]: http://www.in.tu-clausthal.de/fileadmin/homes/techreports/ifi1001hammer.pdf "Hammer, B., and Hasenfuss, A. (2010). Topographic Mapping of Large Dissimilarity Data Sets. Neural Computation, 22(9). pp. 2229-2284. doi: 10.1162/NECO_a_00012"
