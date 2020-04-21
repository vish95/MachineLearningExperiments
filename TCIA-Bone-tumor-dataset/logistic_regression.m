%%Logistic regression for TCIA bone tumor dataset- using function
%%

%data set is labelled with assumption that non-viable tumor is considered
%as a positive tumor

clear all;

%loading pre-shuffled data
load('extracted_data_shuffled.mat');

%split data into train test and validation 
split_percent=[60 20 20];


%we use pre-shuffled data
train_size=round((split_percent(1)/(100))*size(shuffle_data,1));
test_size=round((split_percent(2)/(100))*size(shuffle_data,1));
train_set = shuffle_data(1:train_size,:);
test_set=shuffle_data(train_size+1:train_size+test_size,:);
validation_set= shuffle_data(train_size+test_size:size(shuffle_data,1),:);


%take labels and index column off
train_setf = train_set;
train_setf(:,[1 end])=[];

test_setf = test_set;
test_setf(:,[1 end])=[];

validation_setf = validation_set;
validation_setf(:,[1 end])=[];


[m, n] = size(train_setf);

% Add intercept term to train_set and X_test
X = [ones(m, 1) train_setf];

% Initialize fitting parameters
%initial_theta = zeros(n + 1, 1);
y_train=train_set(:,end);% using extracted labels
% Compute and display initial cost and gradient
%[cost, grad] = costFunction(initial_theta, X, y_train);
initial_theta=zeros(size(X,2),1);
%theta=initial_theta;
lambda=5000;%
[cost, grad] =costFunctionReg(initial_theta, X, y_train, lambda);

disp(cost);
%% ============= Part 3: Optimizing using fminunc  =============


%  Set options for fminunc, some params need to be varied for optimum
%  result
options = optimset('Display','iter','GradObj', 'on', 'MaxIter', 1000);

[theta, cost] = ...
	fminunc(@(t)(costFunctionReg(t, X, y_train,lambda)), initial_theta, options);


%%predict on testing set and tune lambda parameter
test_set_size=size(test_setf,1);
test_result=zeros(test_set_size,1);
y_test=test_set(:,end);
%add ones to test set
X_test = [ones(test_set_size,1) test_setf];
correct_count=0;
for i=1:test_set_size
   pred=sigmoid(theta * X_test(i,:));
   if pred > 0.5
       test_result(i)=1;
   end
   if y_test(i)==test_result(i)
       correct_count=correct_count+1;
   end
end

%f score can be computed optionally but dataset isnt skewed
test_accuracy= (correct_count/test_set_size) *100;
disp(test_accuracy);

v_size=size(validation_setf,1);
val_result=zeros(v_size,1);
X_v = [ones(v_size,1) validation_setf];
y_val= validation_set(:,end);
correct_count=0;
for i=1:v_size
   pred=sigmoid(theta * X_v(i,:));
   if pred > 0.5
       val_result(i)=1;
   end
   if y_val(i) == val_result(i)
       correct_count=correct_count+1;
   end
end
valid_accuracy= (correct_count/test_set_size) *100;
disp(valid_accuracy);