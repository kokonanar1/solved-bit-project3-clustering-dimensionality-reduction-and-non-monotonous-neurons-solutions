Download Link: https://assignmentchef.com/product/solved-bit-project3-clustering-dimensionality-reduction-and-non-monotonous-neurons-solutions
<br>
<strong>task 3.1: fun with k-means clustering: </strong>Download the 2D data in the file data-clustering-1.csv and plot it; the result should look something

like this

Next, implement

<ul>

 <li>Lloyd’s algorithm for <em>k</em>-means clustering (e.g. simply using scipy)</li>

 <li>Hartigan’s algorithm for <em>k</em>-means clustering</li>

 <li>MacQueen’s algorithm for <em>k</em>-means clustering</li>

</ul>

For <em>k </em>= 3, run each algorithm on the above data and plot your results. In fact, run each of them several times and look at the results. What do you observe? Are the results always the same or do they vary from run to run?

Measure the run times of each of your implementations (run them each at least 10 times and determine their average run times). What do you observe?

<strong>task 3.2: spectral clustering: </strong>Download data-clustering-2.csv and plot the 2D data in this file; the result should look something like this

Set <em>k </em>= 2 and apply the <em>k</em>-means algorithms you implemented in the previous task to this data. What do you observe?

Next, implement <em>spectral clustering</em>. Proceed as follows: Let

be the given data. First, compute an <em>n </em>× <em>n </em><em>similarity matrix S </em>where

<em>S<sub>ij </sub></em>= <em><sub>e</sub></em>−<em>β</em>k<em>x</em><em>i</em>−<em>x</em><em>j</em>k<sup>2</sup>

and then compute the <em>Laplacian matrix L </em>= <em>D </em>− <em>S </em>where the diagonal matrix <em>D </em>is given by

otherwise

Note that row <em>i </em>in <em>L </em>can be understood as a feature vector <em>f</em>(<em>x</em><em><sub>i</sub></em>) for data point <em>x</em><em><sub>i</sub></em>. Next, compute the eigenvalues <em>λ<sub>i </sub></em>and eigenvectors <em>u</em><em><sub>i </sub></em>of <em>L </em>and sort them in descending order. That is, let <em>λ<sub>n </sub></em>denote the largest eigenvalue and <em>u</em><em><sub>n </sub></em>denote the corresponding eigenvector.

The eigenvector <em>u</em><sub>2 </sub>that corresponds to the second smallest eigenvalue <em>λ</em><sub>2 </sub>is called the <em>Fiedler vector </em>and is of significance in clustering. You will find that some of its entries are greater than 0 and some are less than 0. To cluster the given data into two clusters <em>C</em><sub>1 </sub>and <em>C</em><sub>2</sub>, do the following: If entry <em>i </em>of <em>u</em><sub>2 </sub>is greater than 0 assign <em>x</em><em><sub>i </sub></em>to cluster <em>C</em><sub>1</sub>, it it is less than zero, assign <em>x</em><em><sub>i </sub></em>to cluster <em>C</em><sub>2</sub>.

Set <em>β </em>to some value (say <em>β </em>= 1), cluster the data as described, and plot your results. What do you observe?

<strong>task 3.3: dimensionality reduction: </strong>The file data-dimred-X.csv contains a 500 × 150 data matrix <em>X</em>, that is, 150 data vectors <em>x</em><em><sub>i </sub></em>∈ R<sup>500</sup>.

In fact, these data vectors are from three classes and you can find the corresponding class labels <em>y<sub>i </sub></em>∈ {1<em>,</em>2<em>,</em>3} in the file data-dimred-y.csv.

The goal of this task is to explore mappings R<sup>500 </sup>→ R<sup>2 </sup>that allow us to visualize (plot) high-dimensional data.

First of all, perform dimensionality reduction using <em>PCA</em>. That is, normalize the data in <em>X </em>to zero mean and compute the eigen-decomposition of the corresponding covariance matrix. Then, use the two eigenvectors <em>u</em><sub>1 </sub>and <em>u</em><sub>2 </sub>of the two largest eigenvalues to project the (normalized!) data into R<sup>2</sup>. What do you observe?

Second of all, perform dimensionality reduction using<em>multiclass LDA </em>(as discussed in lecture 14). To this end, make use of the the fact that the data in <em>X </em>are from three classes. Compute the within class scatter matrix <em>S</em><em><sub>W </sub></em>and the between class scatter matrix <em>S</em><em><sub>B </sub></em>and then the eigen-decomposition of the matrix <em>S</em><em><sub>W</sub></em><sup>−1</sup><em>S</em><em><sub>B</sub></em>. Again, use the two eigenvectors <em>u</em><sub>1 </sub>and <em>u</em><sub>2 </sub>of the two largest eigenvalues to project the data into R<sup>2</sup>. What do you observe? Does the result differ from the one you obtained via PCA?

What if you project the data from R<sup>500 </sup>to R<sup>3</sup>? For both approaches, this can be accomplished using the first three eigenvectors and creating 3D plots. <strong>task 3.4: non-monotonous neurons: </strong>The two files xor-X.csv and xor-y.csv contain data points <em>x</em><em><sub>i </sub></em>∈ R<sup>2 </sup>and label values which when plotted appropriately should lead to a picture like this

Note that XOR problems like this pose nonlinear classification problems, because there is no single hyperplane that would separate the blue from the orange dots. XOR problems are therefore famously used to prove the limitations of a single perceptron

where <em>f </em>is a monotonous activation function such as

<em>.</em>

However, this limitation is a historical artifact, because monotonous activation functions are a persistent <em>meme </em>that arose in the 1940s. Yet, there is nothing that would prevent us from considering more flexible neural networks composed of neurons with non-monotonous activations.

<strong>Mandatory sub-task: </strong>Given the above data, train a non-monotonous neuron

where

<em>.</em>

In order to do so, perform gradient descend over the loss function

<em>.</em>

That is, randomly initialize <em>w</em><sub>0 </sub>∈ R<sup>2 </sup>and <em>θ</em><sub>0 </sub>∈ R and then iteratively compute the updates

<em>∂E w</em><em>t</em>+1 = <em>w</em><em>t </em>− <em>η</em><em>w</em>

<em>∂w</em>

<strong>Note: </strong>Good choices for the step sizes are <em>η<sub>θ </sub></em>= 0<em>.</em>001 and <em>η<sub>w </sub></em>= 0<em>.</em>005, but you are encourages to experiment with these parameters and see how they influence the behavior and outcome of the training procedure.

If all goes well, and say you also implement a function that can visualize a classifier, you should observe something like this

result obtained from <em>w</em><sub>0</sub><em>,θ</em><sub>0                                             </sub>result obtained from <em>w</em><sub>50</sub><em>,θ</em><sub>50</sub>

<strong>Voluntary sub-task: </strong>If you want to impress your professor, then also train a kernel SVM on the above data using a polynomial kernel

<em>.</em>

<strong>task 3.5: exploring numerical instabilities: </strong>This task revisits task 2.1 and is not such much a task in itself but an eye opener! The goal is to raise awareness for the fact that doing math on digital computers may lead to unreliable results! <strong>Everybody, i.e. every member of each team, must do it!</strong>

Download the file whData.dat, remove the outliers and collect the remaining height and weight data in two numpy arrays hgt and wgt and fit a 10-th order polynomial. Use the following code:

import numpy as np import numpy.linalg as la import numpy.polynomial.polynomial as poly import matplotlib.pyplot as plt

hgt = … wgt = …

xmin = hgt.min()-15 xmax = hgt.max()+15 ymin = wgt.min()-15 ymax = wgt.max()+15

def plot_data_and_fit(h, w, x, y): plt.plot(h, w, ’ko’, x, y, ’r-’) plt.xlim(xmin,xmax) plt.ylim(ymin,ymax) plt.show()

def trsf(x): return x / 100.

n = 10

x = np.linspace(xmin, xmax, 100)

<em># method 1:</em>

<em># regression using ployfit </em>c = poly.polyfit(hgt, wgt, n) y = poly.polyval(x, c) plot_data_and_fit(hgt, wgt, x, y)

<em># method 2:</em>

<em># regression using the Vandermonde matrix and pinv</em>

X = poly.polyvander(hgt, n)

c = np.dot(la.pinv(X), wgt) y = np.dot(poly.polyvander(x,n), c) plot_data_and_fit(hgt, wgt, x, y)

<em># method 3:</em>

<em># regression using the Vandermonde matrix and lstsq </em>X = poly.polyvander(hgt, n) c = la.lstsq(X, wgt)[0] y = np.dot(poly.polyvander(x,n), c) plot_data_and_fit(hgt, wgt, x, y)

<em># method 4:</em>

<em># regression on transformed data using the Vandermonde</em>

<em># matrix and either pinv or lstsq </em>X = poly.polyvander(trsf(hgt), n) c = np.dot(la.pinv(X), wgt) y = np.dot(poly.polyvander(trsf(x),n), c) plot_data_and_fit(hgt, wgt, x, y)

What is going on here? Report what you observe! Think about what this implies! What if you were working in aerospace engineering where sloppy code and careless and untested implementations could have catastrophic results . . .