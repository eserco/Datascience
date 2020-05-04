% Genetic algorithm for the selection of the best subset of features
% Course: Introduction to Data Science
% Author: George Azzopardi
% Date:   September, 2018

function bestchromosome = myGeneticAlgorithm(features,labels)
% features = A matrix of independent variables
% labels = A vector that contains the labels for each rows in matrix 
% features

nchroms       = 100; % number of chromosomes
nepochs       = 10;  % number of epochs
nparentsratio = 0.2; % portion of elite list
mutateprob    = 0.1; % probability to mutate a bit in a chromosome

% Create figure that shows the progress of the genetic algorithm
figure;hold on;
title('Feature Selection with Genetic Algorithm');
colorlist = jet(nepochs);

% Convert labels, which can be in string format, to numeric.
[lbls,h] = grp2idx(labels);

% Iterate through all epochs
for epoch = 1:nepochs
    fprintf('epoch %d of %d\n',epoch,nepochs);
    if epoch == 1
        % generate the intial popultion of chromosome with randomly
        % assigned bits
        pop = generateInitialPopulation(nchroms,size(features,2));        
    else
        % generate a new population by creating offspring from the best
        % performing chromosome (or parents)
        pop = getnewpopulation(pop,score,nparentsratio,mutateprob);
    end    
    pop = logical(pop);
    
    % Compute the fitness score for each chromosome
    score = zeros(1,nchroms);
    for i = 1:nchroms
        score(i) = getScore(pop(i,:),features,lbls);    
    end    
    
    % Plot the scores to visualize the progress
    plot(sort(score,'descend'),'color',colorlist(epoch,:));
    xlabel('Chromosome');
    ylabel('Fitness Score');
    legendList{epoch} = sprintf('Epoch %d',epoch);
    legend(legendList);
    drawnow;
end

% Return the chromosome with the maximum fitness score
[~,mxind] = max(score);
bestchromosome = pop(mxind,:);

function newpop = getnewpopulation(pop,score,nparentsratio,mutateprob)
% Generate a new population by first selecting the best performing
% chromosomes from the given pop matix, and subsequently generate new 
% offspring chromosomes from randomly selected pairs of parent chromosomes.

% Step 1. Write code to select the top performing chromosomes. Use 
% nparentsratio to calculate how many parents you need. If pop has 100 rows
% and nparentsration is 0.2, then you have to select the top performing 20
% chromosomes

% Combine scores and pop into a single matrix
nchroms = size(pop,1);
features = size(pop,2);
popScore = zeros(nchroms, features+1);
popScore(:,1:features) = pop;
popScore(:,features+1) = score;

% Sort the population in descending score order
sortedPop = sortrows(popScore,-(features+1));

% Get a round number of needed parents
numberOfParents = ceil(nparentsratio * nchroms);

% Select the best parents
parents = sortedPop(1:numberOfParents,1:features);

% Step 2. Iterate until a new population is filled. Using the above
% example, you need to iterate 80 times. In each iteration create a new
% offspring chromosome from two randomly selected parent chromosomes. Use
% the function getOffSpring to generate a new offspring.

% Make a matrix to hold the new population and fill it with the known
% parents
newPopulation = zeros(nchroms, features);
newPopulation(1:numberOfParents,:) = parents;

% Loop through the rest of the newPopulation and fill it with offspring
% from the parents
for i = numberOfParents:nchroms
    parent1 = parents(randi([1 numberOfParents],1,1),:);
    parent2 = parents(randi([1 numberOfParents],1,1),:);
    newPopulation(i,:) = getOffSpring(parent1, parent2, mutateprob);    
end

newpop = newPopulation;

function offspring = getOffSpring(parent1,parent2,mutateprob)
% Generate an offpsring from parent1 and parent2 and mutate the bits by
% using the probability mutateprob.

% Step 1. Write code that generates one offspring from the given two parents

% Initialize values
features = size(parent1,2);

% Fill the offspring with random bits
offspring = zeros(1, features);
for i = 1:features
   offspring(i) = randi([0 1],1,1);
end

% Choose a random part of the parents' features
beginPoint = randi([1 features],1,1);
endPoint = randi([beginPoint features],1,1);
parent1Features = parent1(:,beginPoint:endPoint);
parent2Features = parent2(:,beginPoint:endPoint);

% Begin creating offspring based on parents
numberOfNeededFeatures = size(parent1Features,2);
offspringBits = zeros(1,numberOfNeededFeatures);
for i = 1:numberOfNeededFeatures
    parent1Bit = parent1Features(:,i);
    parent2Bit = parent2Features(:,i);    
    if (parent1Bit == parent2Bit)
        offspringBits(i) = parent1Bit;
    end
end

% Replace the chosen features
offspring(:,beginPoint:endPoint) = offspringBits;

% Step 2. Write code to mutate some bits with given mutation probability mutateprob
mutationValue = ceil(mutateprob * 100);

% For all the features
for i = 1:features
    % If the chance of mutation has been ment
    if (randi([0 100],1,1) <= mutationValue)
       % Invert the values
       if (offspring(i) == 1)
           offspring(i) = 0;
       elseif (offspring(i) == 0)
           offspring(i) = 1;
       end
    end
end

function score = getScore(chromosome,train_feats,labels)
% Compute the fitness score using 2-fold cross validation and KNN
% classifier

cv = cvpartition(labels,'Kfold',2);
for i = 1:cv.NumTestSets    
    knn = fitcknn(train_feats(cv.training(i),chromosome),labels(cv.training(i)));
    c = predict(knn,train_feats(cv.test(i),chromosome));
    acc(i) = sum(c == labels(cv.test(i)))/numel(c);
end
meanacc = mean(acc);
score = (10^4 * meanacc) + (0.4 * sum(chromosome == 0));

function pop = generateInitialPopulation(n,ndim)
% Generate the initial population of chromosomes with random bits

pop = zeros(n,ndim);

pop(1,:) = ones(1,ndim);
for i = 2:n    
    pop(i,randperm(ndim,mod(i,round(ndim/2))+1)) = 1;
end
