% clear;
% 
% data = load('ex1data1.txt');
% X = data(:, 1); y = data(:, 2);
% m = length(y); % number of training examples
% 
% X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
% theta = zeros(2, 1); % initialize fitting parameters
% 
% % Some gradient descent settings
% iterations = 1500;
% alpha = 0.01;
% num_iters=iterations;

%delete later

function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    h_th=zeros(m,1);

    for i=1:m
        h_th(i,1)=theta(1)*X(i,1)+theta(2)*X(i,2);
    end
    
    diff=h_th-y;
    
    temp1=theta(1)-alpha*(1/m)*diff'*X(:,1);     %quicker scalar op to sum (h_th-y) & x
    temp2=theta(2)-alpha*(1/m)*diff'*X(:,2);    

    theta(1)=temp1;
    theta(2)=temp2;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
