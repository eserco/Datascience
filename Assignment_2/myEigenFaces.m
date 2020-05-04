% Application of face recognition to demonstrate the effectiveness of PCA
% Coure Introduction to Data Science
% Author: George Azzopardi - September 2018
function accuracy = myEigenFaces(KList)

% directory names of image data
trainingFolder = 'data/training-set/';
testingFolder = 'data/testing-set/';

% Convert training and test data into column vectors
[trainingFilenameList, trainingData, trainingimgs] = getData(trainingFolder);
[testingFilenameList, testingData, testingimgs] = getData(testingFolder);

% Show training and test data
% figure; imagesc(cell2mat(reshape(trainingimgs,10,20))); colormap(gray); axis image; axis off; title('Training Data');
% figure; imagesc(cell2mat(reshape(testingimgs,10,7))); colormap(gray); axis image; axis off; title('Test Data');

% Compute mean face
meanTrainingFace = mean(trainingData,2);

% show training mean face
% figure;imagesc(reshape(meanTrainingFace',28,23));colormap(gray);axis image;

% Compute difference images by subtracting the mean face from all training
% images
A = trainingData - repmat(meanTrainingFace,1,size(trainingData,2));

% Determine the principal components and eigenvalues of the matrix A with
% the function that you implemented
[pc, eigenvalues] = mypca(A);

% Show the eigen vectors as images. These are called eigen faces
% %%no, the slides say that the eigenfaces are the u_i vectors,
% that is A*v_i which are the referred to eigenvectors. Now, this actually
% makes sense since the eigenvectors of the covariance matrix do not have
% the proper shape for the image reshape.
% showEigenFaces(A*pc);

% Reconstructs the first training image as a linear combination of the first 10 eigen faces
% again... assuming the U matrix but using pc's in original code
U = normalize(A*pc);
showReconstruction(U,trainingimgs{1},meanTrainingFace,200);

% If KList is not provided as an input parameter then we choose it to be
% the one such that the total variance explains at least 90% of all the data 
if nargin == 0
    Z = cumsum(eigenvalues)./sum(eigenvalues);
    KList = 1:find(Z > 0.9,1);
end

% Compute classification accuracy using different number of the top eigen vectors.
idx = 1;
accuracy = zeros(1,numel(KList));
[labelsid, mapid] = grp2idx(trainingFilenameList);
for K = KList
    trainingVectors = zeros(size(trainingData));
    testingVectors = zeros(size(testingData));
    %project trainingdata onto the training principal components
    %the testdata is projected ontot he same axes, which seems to be more
    %justified than using testing pc's, we should not use its information.
    for im_idx = 1:size(trainingData,2)
        trainingVectors(:, im_idx) = reconstructFace(U, trainingData(:,im_idx), meanTrainingFace, K);
    end
    for im_idx = 1:size(testingData,2)
        testingVectors(:, im_idx) = reconstructFace(U, testingData(:,im_idx), meanTrainingFace, K);
    end
    
    % Compute accuracy
    accuracy(idx) = classify(testingVectors,trainingVectors,trainingFilenameList,testingFilenameList);
            
    idx = idx + 1;
end

% Show plot of accuracy rates with the considered varying number of K principal components
if numel(KList) > 1
    figure;plot(accuracy,'r.-');
    hold on;
    plot(Z(KList),'b.-')
    xlabel('# Top Eigen Vectors considered');     
    legend('Accuracy Rate','Percentage of total variance');
end

function accuracy = classify(testingVectors,trainingVectors,trainingFilenameList,testingFilenameList)
% Compute the accuracy rate of the test data using 1-Nearest Neighbour
d = pdist2(testingVectors',trainingVectors');
[~,mnidx] = min(d,[],2);

total = numel(testingFilenameList);
correct = 0;
for i = 1:numel(testingFilenameList)
    [~,testfilename,~] = fileparts(testingFilenameList{i});
    [~,trainingfilename,~] = fileparts(trainingFilenameList{mnidx(i)});
    if strcmp(testfilename(1:4),trainingfilename(1:4))
        correct = correct + 1;
    end
end
accuracy = correct / total;

function recon = reconstructFace(u, orgIm, mu, k)
% Reconstructs an image from a given number of principal components
recon = 0;
for i = 1:k
    recon = recon + (u(:,i)'*(double(orgIm(:))-mu) * u(:,i));    
end
recon = recon + mu;

function showReconstruction(u,orgIm,mu,k)
% Visual a reconstructed face
recon = reconstructFace(u,orgIm,mu,k);
figure;
subplot(1,2,1);imagesc(orgIm);colormap(gray);axis image;axis off;title('Original Image');
subplot(1,2,2);imagesc(reshape(recon,28,23));colormap(gray);axis image;axis off;title('Reconstruced Image');

function showEigenFaces(eigenvectors)
% Visualize the top 20 principal components or eigen faces as are known in
% this application
k = 20;
list = cell(1,k);
for i = 1:k
    list{i} = reshape(eigenvectors(:,i),28,23);
end
figure;imagesc(cell2mat(reshape(list,5,4)));colormap(gray);axis image;axis off;

function [filenameList, dataMatrix, im] = getData(inputFolder)
% Reads data from given folder
d = dir([inputFolder,'*.jpg']);
dataMatrix = [];
filenameList = cell(0);
for i = 1:numel(d)
    filenameList{i} = [inputFolder,d(i).name];
    im{i} = imread(filenameList{i});
    dataMatrix(:,i) = im{i}(:);    
end