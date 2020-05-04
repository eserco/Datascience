72/100

Feedback:
Q1.1 (Eigen): Well Done
Q1.2 (ANOVA):Well Done
Q2.1 (F-statistic): 
Explanation in report is ok
7/10 (point has been increased; initial point was due to the result from submitted code being sensitive to label ordering: see next comment)
The given code is sensitive to number of groups and also ordering of labels: implementation is not optimal since it is dependent on the ordering of labels, and assuming only 3 class class when the expected code was to be more general (so, it should be able to deal with a 2 class or 4 class or more than 4 class problem as well)

Proof that result from the submitted code is sensitive to ordering of labels can be seen in image01.png.

Proof that results from MATLAB’s inbuilt anova1 is not sensitive to label ordering can be seen in image02.png.. This is how one expects the F-statistic computation to be- independent of label ordering.
Q2.2 (PCA): 
Data needs to be normalized and centered before computation of covariance matrix for eig(). Otherwise, features with larger values and bigger variance will dominate (contribute more to the first few principal axes) rather than the features truly significant being contributing to the first few principal axes. In this submission only centering of data has been done. For future assignments related to PCA or covariance, one can brush through the effect of cenetring, standardization over here https://sebastianraschka.com/faq/docs/pca-scaling.html
However, no point has been deducted for not normalizing your data in this part of the assignment
L2norm has not been performed on the PCs
19/20
Q2.3 (Genetic): 13/20
In getnewpop()
You should ensure that the 2 parents cannot be the same set of chromosomes. 
In getoffspring()
Have to ensure that the offspring cannot be an entire copy of parent. Think about situation where crossover = 1 or length of parent

Q2.4 (Application):
8/10 Does not compile? Attempted implementation is rather inefficient due to the use of loops. Try to use matrix/vector operations instead. No plot shown in the report.
5/20 The outcome of the algorithms are not shown. When discussing results, include the figures. Advantages/disadvantages not really mentioned for each approach.
