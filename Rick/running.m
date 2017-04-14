% Authored by Rick~
clc;
clear all;
img = imread('IM-0001-0001.jpg');
img = rgb2gray(img);
img = img(250:290,250:290);

N =7;
[constants,params,net,init_grad]  = Init(img,N );
[ x_output,net_forwarded ] = Forward( N,constants,params,net);
[ E,grad ] = Back(img,constants,params,init_grad,net_forwarded);
back_func = @(p)Back( img,constants,params,init_grad,net_forwarded);

grad = real(grad);
options = struct('GradObj','on','Display','iter','LargeScale','off','HessUpdate','lbfgs','InitialHessType','identity','GoalsExactAchieve',0);
[opt_grad,cost] = fminlbfgs(back_func,real(cell2mat(params.rho(1:7))),options);
%rollGrad=RollGradient(opt_grad,init_grad);
for i = 1:7
    params.rho(i,1) = {opt_grad(i)};
end
%params.rho(8,1) = par
[ x_output2,net_forwarded2 ] = Forward( N,constants,params,net);
k = reshape(cell2mat(x_output2),[size(img,1),size(img,1)]);
imshow(real(k));
hold on;
imshow(real(img));
