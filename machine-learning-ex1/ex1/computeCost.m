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

%delete later

function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
% J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

h_th=zeros(m,1);

for i=1:m
    h_th(i,1)=theta(1)*X(i,1)+theta(2)*X(i,2);
end

J=(1/(2*m))*sum((h_th-y).^2);

% =========================================================================

end
