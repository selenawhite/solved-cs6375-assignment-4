Download Link: https://assignmentchef.com/product/solved-cs6375-assignment-4
<br>
<ol>

 <li><strong>**Support Vector Machines with Synthetic Data**</strong><strong>,. </strong><strong>¶</strong></li>

</ol>

For this problem, we will generate synthetic data for a nonlinear binary classification problem and partition it into training, validation and test sets. Our goal is to understand the behavior of SVMs with Radial-Basis Function (RBF) kernels with different values of C and γ.

<em># DO NOT EDIT THIS FUNCTION; IF YOU WANT TO PLAY AROUND WITH DATA GENERATION, </em>

<em># MAKE A COPY OF THIS FUNCTION AND THEN EDIT</em>

<em>#</em>

<h1>import numpy as np</h1>

<strong>from</strong> <strong>sklearn.datasets</strong> <strong>import</strong> make_moons <strong>from</strong> <strong>sklearn.model_selection</strong> <strong>import</strong> train_test_split <strong>import</strong> <strong>matplotlib.pyplot</strong> <strong>as</strong> <strong>plt </strong><strong>from</strong> <strong>matplotlib.colors</strong> <strong>import</strong> ListedColormap

<strong>def</strong> generate_data(n_samples, tst_frac=0.2, val_frac=0.2):

<em># Generate a non-linear data set</em>

X, y = make_moons(n_samples=n_samples, noise=0.25, random_state=42)




<em># Take a small subset of the data and make it VERY noisy; that is, generate outliers</em>  m = 30

np.random.seed(30)  <em># Deliberately use a different seed</em>

ind = np.random.permutation(n_samples)[:m]

X[ind, :] += np.random.multivariate_normal([0, 0], np.eye(2), (m, ))  y[ind] = 1 – y[ind]

<em># Plot this data</em>

cmap = ListedColormap([‘#b30065’, ‘#178000′])

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors=’k’)




<em># First, we use train_test_split to partition (X, y) into training and test sets</em>

X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=tst_frac,                                                 random_state=42)

<em># Next, we use train_test_split to further partition (X_trn, y_trn) into tra ining and validation sets</em>

X_trn, X_val, y_trn, y_val = train_test_split(X_trn, y_trn, test_size=val_fr ac,                                                 random_state=42)




<strong>return</strong> (X_trn, y_trn), (X_val, y_val), (X_tst, y_tst)

<em>#</em>

<em>#  DO NOT EDIT THIS FUNCTION; IF YOU WANT TO PLAY AROUND WITH VISUALIZATION, </em>

<em>#  MAKE A COPY OF THIS FUNCTION AND THEN EDIT </em>

<em>#</em>

<strong>def</strong> visualize(models, param, X, y):

<em># Initialize plotting</em>  <strong>if</strong> len(models) % 3 == 0:    nrows = len(models) // 3  <strong>else</strong>:    nrows = len(models) // 3 + 1




fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(15, 5.0 * nrows))  cmap = ListedColormap([‘#b30065’, ‘#178000’])

<em># Create a mesh</em>

xMin, xMax = X[:, 0].min() – 1, X[:, 0].max() + 1  yMin, yMax = X[:, 1].min() – 1, X[:, 1].max() + 1  xMesh, yMesh = np.meshgrid(np.arange(xMin, xMax, 0.01),                              np.arange(yMin, yMax, 0.01))

<strong>for</strong> i, (p, clf) <strong>in</strong> enumerate(models.items()):    <em># if i &gt; 0:</em>    <em>#   break</em>    r, c = np.divmod(i, 3)    ax = axes[r, c]

<em># Plot contours</em>

zMesh = clf.decision_function(np.c_[xMesh.ravel(), yMesh.ravel()])    zMesh = zMesh.reshape(xMesh.shape)

ax.contourf(xMesh, yMesh, zMesh, cmap=plt.cm.PiYG, alpha=0.6)

<strong>if</strong> (param == ‘C’ <strong>and</strong> p &gt; 0.0) <strong>or</strong> (param == ‘gamma’):      ax.contour(xMesh, yMesh, zMesh, colors=’k’, levels=[-1, 0, 1],                  alpha=0.5, linestyles=[‘–‘, ‘-‘, ‘–‘])

<em># Plot data</em>

ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors=’k’)           ax.set_title(‘<strong>{0}</strong> = <strong>{1}</strong>‘.format(param, p))

<h1>a.  The effect of the regularization parameter, C</h1>

Complete the Python code snippet below that takes the generated synthetic 2-d data as input and learns nonlinear SVMs. Use scikit-learn’s <a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html">SVC </a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html">(</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html">https://scikit-learn.or</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html">g</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html">/stable/modules/</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html">g</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html">enerated/sklearn.svm.SVC.html) </a>function to learn SVM models with <strong>radial-basis kernels</strong> for fixed γ and various choices of

C ∈ {10<sup>−3</sup>,10<sup>−2 </sup>⋯,1, ⋯ 10<sup>5</sup>}. The value of γ is fixed to γ = <sub>d⋅</sub><u><sup>1</sup></u><sub>σ</sub>X , where d is the data dimension and σ<sub>X</sub> is the standard deviation of the data set X. SVC can automatically use these setting for γ if you pass the argument gamma = ‘scale’ (see documentation for more details).

<strong>Plot</strong>: For each classifier, compute <strong>both</strong> the <strong>training error</strong> and the <strong>validation error</strong>. Plot them together, making sure to label the axes and each curve clearly.

<strong>Discussion</strong>: How do the training error and the validation error change with C? Based on the visualization of the models and their resulting classifiers, how does changing C change the models? Explain in terms of minimizing the SVM’s objective function w<sup>′</sup>w  x<sub>i</sub>,y<sub>i</sub>), where ℓ is the hinge loss for each training example (x<sub>i</sub>,y<sub>i</sub>).

<strong>Final Model Selection</strong>: Use the validation set to select the best the classifier corresponding to the best value, C<sub>best</sub>. Report the accuracy on the <strong>test set</strong> for this selected best SVM model. <em>Note: You should report a single number, your final test set accuracy on the model corresponding to $C</em>{best}$_.

File “&lt;ipython-input-4-8875a1448a41&gt;”, line 17     visualize(models, ‘C’, X_trn, y_trn)

^

IndentationError: expected an indented block

<h1>b.  The effect of the RBF kernel parameter, γ</h1>

Complete the Python code snippet below that takes the generated synthetic 2-d data as input and learns various <a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html">non-linear SVMs. Use scikit-learn’s </a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html">SVC </a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html">(</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html">https://scikit-</a>

<a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html">learn.or</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html">g</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html">/stable/modules/</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html">g</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html">enerated/sklearn.svm.SVC.html)</a> function to learn SVM models with <strong>radial-basis kernels</strong> for fixed C and various choices of γ ∈ {10<sup>−2</sup>,10<sup>−1 </sup>1,10, 10<sup>2 </sup>10<sup>3</sup>}. The value of C is fixed to

C = 10.

<strong>Plot</strong>: For each classifier, compute <strong>both</strong> the <strong>training error</strong> and the <strong>validation error</strong>. Plot them together, making sure to label the axes and each curve clearly.

<strong>Discussion</strong>: How do the training error and the validation error change with γ? Based on the visualization of the

models and their resulting classifiers, how does changing γ change the models? Explain in terms of the functional form of the RBF kernel, κ(x, z) = exp(−γ ⋅ ∥x − z∥<sup>2</sup>)

<strong>Final Model Selection</strong>: Use the validation set to select the best the classifier corresponding to the best value, γ<sub>best</sub>. Report the accuracy on the <strong>test set</strong> for this selected best SVM model. <em>Note: You should report a single number, your final test set accuracy on the model corresponding to $gamma</em>{best}$_.




<ol start="2">

 <li><strong>**Breast Cancer Diagnosis with Support Vector Machines**</strong><strong>, 25 points.</strong></li>

</ol>

<a href="https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic">For this problem, we will use the </a><a href="https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic">Wisconsin Breast Cancer</a>

<a href="https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic">(https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+</a><a href="https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic">(</a><a href="https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic">Dia</a><a href="https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic">g</a><a href="https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic">nostic)</a>) data set, which has already <a href="https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.loadtxt.html">been pre-processed and partitioned into training, validation and test sets. Numpy’s </a><a href="https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.loadtxt.html">loadtxt</a>

<a href="https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.loadtxt.html">(https://docs.scip</a><a href="https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.loadtxt.html">y</a><a href="https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.loadtxt.html">.or</a><a href="https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.loadtxt.html">g</a><a href="https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.loadtxt.html">/doc/nump</a><a href="https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.loadtxt.html">y</a><a href="https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.loadtxt.html">-1.13.0/reference/</a><a href="https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.loadtxt.html">g</a><a href="https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.loadtxt.html">enerated/nump</a><a href="https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.loadtxt.html">y</a><a href="https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.loadtxt.html">.loadtxt.html)</a><a href="https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.loadtxt.html"> comma</a>nd can be used to load CSV files.

Use scikit-learn’s <a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html">SVC </a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html">(</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html">https://scikit-learn.or</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html">g</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html">/stable/modules/</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html">g</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html">enerated/sklearn.svm.SVC.html)</a> function to learn SVM models with <strong>radial-basis kernels</strong> for <strong>each combination</strong> of C ∈ {10<sup>−2</sup>,10<sup>−1</sup>,1,10<sup>1</sup>, ⋯ 10<sup>4</sup>} and γ ∈ {10<sup>−3</sup>,10<sup>−2 </sup>10<sup>−1</sup>,1, 10, 10<sup>2</sup>}. Print the tables corresponding to the training and validation errors.

<strong>Final Model Selection</strong>: Use the validation set to select the best the classifier corresponding to the best parameter values, C<sub>best</sub> and γ<sub>best</sub>. Report the accuracy on the <strong>test set</strong> for this selected best SVM model. <em>Note: You should report a single number, your final test set accuracy on the model corresponding to $C</em>{best} andgamma<em>{best}$</em>.

<ol start="3">

 <li><strong>**Breast Cancer Diagnosis with </strong>k<strong>-Nearest Neighbors**</strong><strong>, </strong></li>

</ol>

<a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html">Use scikit-learn’s </a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html">k-nearest nei</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html">g</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html">hbor </a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html">(</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html">https://scikit-</a>

<a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html">learn.or</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html">g</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html">/stable/modules/</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html">g</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html">enerated/sklearn.nei</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html">g</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html">hbors.KNei</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html">g</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html">hborsClassifier.html)</a> classifier to learn models for Breast Cancer Diagnosis with k ∈ {1, 5, 11, 15, 21}, with the kd-tree algorithm.

<strong>Plot</strong>: For each classifier, compute <strong>both</strong> the <strong>training error</strong> and the <strong>validation error</strong>. Plot them together, making sure to label the axes and each curve clearly.

<strong>Final Model Selection</strong>: Use the validation set to select the best the classifier corresponding to the best parameter value, k<sub>best</sub>. Report the accuracy on the <strong>test set</strong> for this selected best kNN model. <em>Note: You should report a single number, your final test set accuracy on the model corresponding to $k</em>{best}$_.

<strong>Discussion</strong>: Which of these two approaches, SVMs or kNN, would you prefer for this classification task? Explain.