
% x refers to the population size in 10,000s
% y refers to the profit in $10,000s
%

%% Initialization
clear ; close all; clc

%% ======================= Part 1: Plotting =======================
fprintf('Plotting Data ...\n')
data = load('lab1data1.txt');
X = data(:, 1); y = data(:, 2);
m = length(y); % number of training examples

% Plot Data
% Note: You have to complete the code in plotData.m
plotData(X, y);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ============ Linear regression using the normal equation =======
X = [ones(size(X,1),1) data(:,1)];
y = data(:, 2);
theta = linearReg(X, y);
% print theta to screen
fprintf('Theta found by normal equation : \n');
fprintf('%f\n', theta);

% Plot the linear fit
hold on; % keep previous plot visible
plot(X(:,2), X*theta, '-')
legend('Training data', 'Linear regression')
hold off % don't overlay any more plots on this figure

fprintf('Prediction using the previous values of theta...\n');
% Predict values for population sizes of 35,000 and 70,000
predict1 = [1, 3.5] *theta;
fprintf('For population = 35,000, we predict a profit of %f\n',...
    predict1*10000);
predict2 = [1, 7] * theta;
fprintf('For population = 70,000, we predict a profit of %f\n',...
    predict2*10000);

fprintf('Program paused. Press enter to continue.\n\n');
pause;

%% ============ Linear regression with multiple variables =========

data = load('lab1data2.txt');
X = data(:, 1:2); y = data(:, 3);
m = length(y); % number of training examples

% Finding theta using the normal equation
X = [ones(size(X,1),1) data(:, 1:2)];
theta = linearReg(X, y);
fprintf('Theta for multiple variables : \n');
fprintf('%f\n', theta);

fprintf('Prediction using the previous values of theta...\n');
Xe = [1 1650 3];
price = Xe * theta;
fprintf('A house with 1650 square feet and 3 bedrooms costs : %f \n', price);

fprintf('Program paused. Press enter to continue.\n\n');
pause;
% Normalization part
X = data(:, 1:2);
[Xn, mu, sigma] = featureNormalize(X);
X = [ones(size(X,1),1) Xn(:,:)];

theta2 = linearReg(X, y);

fprintf('Theta for multiple variables (with normalization): \n');
fprintf('%f\n', theta2);

fprintf('Prediction using the previous values of theta...\n');
Xe = featureNormalize([1650 3]);
Xe = [1 Xe];
price = Xe * theta2;
fprintf('A house with 1650 square feet and 3 bedroom costs : %f \n', price);

fprintf('Program paused. Press enter to continue.\n\n');
pause;
%% ================ Gradient Descent ================
fprintf('Gradient descent\n');

% Choose some alpha value
alpha = 1;
num_iters = 50;

fprintf('I chose a value of alpha = %f because the algorithm converges quickly \nand if I take alpha = %f then it diverges!!\n\n', 1,3);

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

fprintf('Prediction using the previous values of theta...\n');

price = 0; 
X = [1650 3];
[X mu sigma] = featureNormalize(X);
X = [1 X];
price = X * theta;
% ============================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 bedrooms house ' ...
         '(using gradient descent):\n $%f\n'], price);
     
fprintf('The price predicted using gradient descent \nis exactly the same as the one predicted using the normal equation.\n\n');

fprintf('-----------End of script------------\n');
