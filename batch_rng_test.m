% Test script for Batch relational neural gas: We create a dataset of
% gaussian clusters, which should be easily recognizable, and apply the
% java implementation as well as the reference implementation of Hasenfuss.

clear all;
clear java;
close all;
clc;

%%

N = 90;
K = 3;
dims = 2;
epochs = 10;

cluster_size = N/K;

X = zeros(N,dims);

for k=1:K
    mu = [k , repmat(0,1,dims-1)];
    sigma = 0.01*eye(dims);
    X((k-1)*cluster_size+1:k*cluster_size,:) = mvnrnd(mu,sigma,cluster_size);
end

figure(1);
plot(X(:,1),X(:,2),'rx');
axis([0,4,-2,2]);

% create distance matrix
D = squareform(pdist(X)).^2;
figure(2);
imagesc(D);
colorbar();

%% cluster with Hasenfuss' script

addpath ~/SVN/fit/Matlab/Matlab-Metric/functions/relational_methods/toolbox;
addpath ~/SVN/fit/Matlab/Matlab-Metric/functions/relational_methods/toolbox_relational;
addpath ~/SVN/fit/Matlab/Matlab-Metric/functions/relational_methods/relational_bng;

%%

[protos, D_protos_to_data, errors] = bng_relational(D, K, epochs);

% show the assignments

[~, assignments] = min(D_protos_to_data);

figure(3);
colors = 'rbgkcm';
proto_positions = protos * X;
for k=1:K
    % get datapoints assigned to this cluster
    X_cluster = X(assignments == k,:);
    % plot the points
    plot(X_cluster(:,1), X_cluster(:,2), [colors(k) 'x']);
    hold on;
    % plot the prototype
    plot(proto_positions(k,1), proto_positions(k,2), [colors(k) 'd'], 'MarkerSize', 10);
end

title('Reference Implementation');
axis([0,4,-2,2]);

hold off;

%% cluster with my java implementation

javaaddpath 'target/rng-0.8.0.jar';

%%

calc = de.citec.tcs.rng.BatchRNGCalculator();
calc.setNumberOfEpochs(epochs);
calc.setNumberOfPrototypes(K);

model = calc.calculate(D);

protos = model.getConvexCoefficients();
assignments = de.citec.tcs.rng.RNGModelFunctions.getAssignments(model)+1;

figure(4);
colors = 'rbgkcm';
proto_positions = protos * X;
for k=1:K
    % get datapoints assigned to this cluster
    X_cluster = X(assignments == k,:);
    % plot the points
    plot(X_cluster(:,1), X_cluster(:,2), [colors(k) 'x']);
    hold on;
    % plot the prototype
    plot(proto_positions(k,1), proto_positions(k,2), [colors(k) 'd'], 'MarkerSize', 10);
end

title('Java Implementation');
axis([0,4,-2,2]);

hold off;