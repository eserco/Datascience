%Authors: Eser Comak, Carlos Huerta
%This function will run PCA
%Function assumes the dimensions are laid out column wise and the samples
%are rows
function [pc, eigenvalues] = mypca(A)
 
 %take the mean for each dimension
 dimensionMean = mean(A); 
 
 %duplicate the dimensionMean vector into NxN matrix
 meanMatrix = repmat(dimensionMean,size(A,1),1);
 
 %substract meanMatrix from initial matrix to centralize data points
 centeredA = A - meanMatrix;
 
 %compute covariance matrix
 covarianceA = cov(centeredA);
 
 %compute eigen vectors and values
 [eigenvectors, eigenvalues] =eig(covarianceA);
 
 %display the results
 disp('The eigenvalues are: ');
 disp(eigenvalues);
 
 disp('The eigenvectors are: ');
 disp(eigenvectors);

 %store the results in a vector from (extract from the diagonal)
 eigenvalues = diag(eigenvalues);
 
 %sort eigen values in descending order
 [mx,srtidx] = sort(eigenvalues,'descend');
 eigenvalues = eigenvalues(srtidx);
 
 %Display for debugging
 disp('The sorted eigenvalues are: ');
 disp(eigenvalues);
 
 pc = eigenvectors(:,srtidx);
 
 %Display for debugging
 disp('Thesrtidx is: ');
 disp(srtidx);
 
end
