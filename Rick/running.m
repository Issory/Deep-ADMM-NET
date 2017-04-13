% Authored by Rick~
clc;
clear all;
img = imread('IM-0001-0001.jpg');
img = rgb2gray(img);
img = img(256:258,256:258);

N =7;
[constants,params,net,init_grad]  = Init(img,N );
[ x_output,net_forwarded ] = Forward( N,constants,params,net);
[ E,grad ] = Back(img,constants,params,init_grad,net_forwarded);
back_func = @(p)Back( img,constants,params,init_grad,net);

grad = real(grad);
options = struct('GradObj','on','Display','iter','LargeScale','off','HessUpdate','lbfgs','InitialHessType','identity','GoalsExactAchieve',0);
[opt_grad,cost] = fminunc(back_func,cell2mat(init_grad.rho_grad),options);
%rollGrad=RollGradient(opt_grad,init_grad);