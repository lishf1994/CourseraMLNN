function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
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

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

%size(X)

%pause;

%---add bias parameter to X-----
X = [ones(m,1) X];

%size(X)

%pause;

z1= X* Theta1';
a1 = sigmoid (z1);

%size(a1)



sizea1 = size(a1,1);
a1 = [ones(sizea1,1) a1];

z2 = a1* Theta2';
a2 = sigmoid(z2);

h=a2;

%======calculate the h, check the size=====
%size(h);


yClasses = ones(size(y,1), num_labels);
for i = 1:num_labels
  compareColumn = y==i;
  yClasses(:,i) = compareColumn;
end



%size(yClasses)


  
for i = 1:m
  pt1 = yClasses(i,:) * log(h(i,:)');
  pt2 = (1-yClasses(i,:)) * log(1-(h(i,:))');
  J = J + pt1 + pt2;  
end

J = -J/m;

%=========================================================================
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
D2 = zeros(size(Theta2));
 D1 = zeros(size(Theta1));
  
  
for i=1:m
  zTwoi = X(i,:) * Theta1';
  
  %fprintf('sizeofzTwoi_1'); 
  %size(zTwoi)
  %pause;
  
  aTwoi = sigmoid(zTwoi);
  
  %fprintf('size of aTwoi = sigmoid(zTwoi)');
  %size(aTwoi)
  %pause;
  
  size_aTwoi = size(aTwoi,1);
  aTwoi = [ones(size_aTwoi,1) aTwoi];
  
  %fprintf('size of aTwoi after adding');
  %size(aTwoi)
  %pause;
  

  
  zThreei = aTwoi * Theta2';
  aThreei = sigmoid (zThreei);
  
  delta_3 = aThreei - yClasses (i,:); %calculate delta_3 for each X (each row of matrix X)
  
  %calculate delta_2 for each X (each row of matrix X)
  delta_2 = ((delta_3 * Theta2)(2:end)) .* sigmoidGradient(zTwoi);
 

 % D1 and D2 dimensions are fixed.  So each time, the values of each position in D1 and D2 are accumulating.

  D2 = D2 + delta_3' * aTwoi;
  
  D1 = D1 + delta_2' * X(i,:);
  % note that the portion after the plus sign already fits the dimension of D1/D2
  % so the loop is only to add the numbers incrementally.
   
end 

Theta1_grad = 1/m* D1;



Theta2_grad = 1/m* D2;


 



%=============================================================================
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
Theta1Reg = Theta1;
Theta1Reg(:,1)=0;


Theta2Reg = Theta2;
Theta2Reg(:,1)=0;


RegSum=0;

Theta1RegCol = size(Theta1Reg,2);
Theta2RegCol = size(Theta2Reg,2);

%Theta1Reg(:,Theta1RegCol) .^2


for i = 1:Theta1RegCol
  columnSum = sum(Theta1Reg(:,i) .^2);
  RegSum = RegSum + columnSum;
end



for i = 1:Theta2RegCol
  columnSum2 = sum(Theta2Reg(:,i) .^2);
  RegSum = RegSum + columnSum2;
end



J = J + lambda/2/m*RegSum;



Theta1_grad = Theta1_grad+ lambda/m*Theta1Reg;
Theta2_grad = Theta2_grad+ lambda/m*Theta2Reg;









% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
