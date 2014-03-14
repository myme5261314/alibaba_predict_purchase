function [ output_args ] = NNTrain( input_args )
%NNTRAIN Summary of this function goes here
%   Detailed explanation goes here
input_layer_size = 5;
hidden_layer_size = 64*2;
output_layer_size = 1;
lambda = 1;

%   Calculate the num of training case.
if ~exist('dataset.mat','file')
    [train_set, test_set] = splitTrainTestSet();
    save('dataset.mat', 'train_set', 'test_set');
else
    load('dataset.mat');
end
if ~exist('featureLabel.mat', 'file')
    [X, y] = extractGroupFeature(train_set);
    [testX, testy] = extractGroupFeature(test_set);
    save('featureLabel.mat', 'X', 'y', 'testX', 'testy');
else
    load('featureLabel.mat');
end
% pre-process.
X = bsxfun(@minus, X, mean(X));
X = bsxfun(@rdivide, X, std(X));
y = y/100;
testX = bsxfun(@minus, testX, mean(testX));
testX = bsxfun(@rdivide, testX, std(testX));
testy = y/100;

%   Randomly Initialize the NN weight Matrix.
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, output_layer_size);

%   Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
clear initial_Theta1 initial_Theta2;

options = optimset('MaxIter', 50);
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   output_layer_size, X, y, lambda);
% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 output_layer_size, (hidden_layer_size + 1));

%   Predict
pred = predict(Theta1, Theta2, X);
pred_threshold = 0.85;
pred(pred>=pred_threshold)=1;
pred(pred<pred_threshold)=0;
tempy = y;
tempy(tempy>=1) = 1;
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == tempy)) * 100);

pred = predict(Theta1, Theta2, testX);
pred_threshold = 0.85;
pred(pred>=pred_threshold)=1;
pred(pred<pred_threshold)=0;
tempy = testy;
tempy(tempy>=1) = 1;
fprintf('\nTest Set Accuracy: %f\n', mean(double(pred == tempy)) * 100);

end




