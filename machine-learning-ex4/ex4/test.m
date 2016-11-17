clc; clear all;

load('params.mat');

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;

Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

a1=[ones(m,1),X];

z2=a1*Theta1';
a2=sigmoid(z2);

a2=[ones(size(a2,1),1) , a2];
z3=a2*Theta2';

h=sigmoid(z3);


for i=1:num_labels

   Y= y==i;

   h_th=h(:,i);

   cost=(-Y'*log(h_th))-(1-Y)'*log(1-h_th);

   J=J+(1/m)*sum(cost(:));

end

reg_param=(lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2)) ...
    +sum(sum(Theta2(:,2:end).^2)));
                        
J=J+reg_param;

% -------------------------------------------------------------

delta1=zeros(size(Theta1));
delta2=zeros(size(Theta2));

    for t=1:m
        
        a1_t=a1(t,:);
        a2_t=a2(t,:);
        h_t=h(t,:);
        
        Y=false(1,num_labels);
        Y(y(t))=1;
        
        del_3=h_t-Y;
        
        del_2=del_3*Theta2.*sigmoidGradient([1,z2(t,:)]);
        
        del_2=del_2(2:end);
        
        delta1=delta1+ del_2' * a1_t;
        delta2=delta2+ del_3' * a2_t;
        
    end
    
    
    Theta1_grad=(1/m)*delta1;
    Theta2_grad=(1/m)*delta2;

    
    






% =========================================================================

% Unroll gradients
% grad = [Theta1_grad(:) ; Theta2_grad(:)];
% grad=0;
