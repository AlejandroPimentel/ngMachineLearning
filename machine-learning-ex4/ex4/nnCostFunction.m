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

X_ = [ones(m, 1) X];
Z2 = sigmoid(X_ * Theta1');
mz = size(Z2, 1);
Z2_ = [ones(mz, 1) Z2];
a3 = Z2_*Theta2';
h_ = sigmoid(a3);
Y = zeros(size(y)(1), num_labels);

for i = 1:num_labels;
  Y(:, i) = y == i;
end

J = sum((1/m) * sum(-log(h_) .* Y - log(1- h_).*(1-Y)));;% + (lambda/(2*m))*altheta' * altheta ;

altheta1 = Theta1;
altheta1(:,1) = 0;
altheta2 = Theta2;
altheta2(:,1) = 0;
altnn_params = [altheta1(:) ; altheta2(:)];


J = J + (lambda/(2*m))* (altnn_params' * altnn_params) ;





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

Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
grad1_ = zeros(size(Theta1));
grad2_ = zeros(size(Theta2));

for i = 1:m;
  
  %Set the input layer’s values (a (1) ) to the t-th training example x (t) .
  %Perform a feedforward pass (Figure 2), computing the activations (z (2) , 
  %a (2) , z (3) , a (3) )
  %for layers 2 and 3. Note that you need to add a +1 term to ensure that
  %the vectors of activations for layers a (1) and a (2) also include the bias
  %unit. In Octave/MATLAB, if as a column vector, adding one corre-
  %sponds to a 1 = [1 ; a 1].
  
  x = X(i,:);
  a1 = [1 x];
  z2 = a1 * Theta1';
  a2 = sigmoid(z2);
  a2 = [1 a2];
  z3 = a2*Theta2';
  a3 = sigmoid(z3);
  
  %% For each output unit k in layer 3 (the output layer), set δ k = (a k − y k ),
  %where y k ∈ {0, 1} indicates whether the current training example be-
  %longs to class k (y k = 1), or if it belongs to a different class (y k = 0).
  %You may find logical arrays helpful for this task (explained in the pre-
  %vious programming exercise).
  
  y_  = y(i) == 1:num_labels;
  delta3 = a3 - y_;
  
  
  %% . For the hidden layer l = 2, set δ (2) = Θ (2) T δ (3) . ∗ g 0 (z (2) )
  
  delta2 = delta3 * Theta2 .* a2 .* (1-a2);
  delta2_ = delta2(2:end);
  %Accumulate the gradient from this example using the following formula.
  %Note that you should skip or remove δ 0 . In Octave/MATLAB,removing δ 0 
  %corresponds to delta 2 = delta 2(2:end).
  
  grad2_ = delta3' * a2;
  grad1_ = delta2_' * a1;
  Theta1_grad = Theta1_grad + grad1_;
  Theta2_grad = Theta2_grad + grad2_;
  
end
  Theta1_grad = Theta1_grad ./ m;
  Theta2_grad = Theta2_grad ./ m;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%












% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
