% Authored by Rick~
clc;
clear all;
img = imread('IM-0001-0001.jpg');
img = rgb2gray(img);
img = img(250:260,250:260);

N =7;
[constants,params,net,init_grad,init_grad_opt]  = Init(img,N );
grad_opt_unroll = UnrollGradient_opt(init_grad_opt);
[ x_output,net_forwarded ] = Forward( N,constants,params,net);

[ E,grad_opt_unroll ] = Back_opt(grad_opt_unroll,img,constants,params,init_grad,net_forwarded);

%back_func = @(grad_opt_unroll)Back_opt( grad_opt_unroll,img,constants,params,init_grad,net_forwarded);
main = @(grad_opt_unroll)main_process(img,N,constants,params,net,init_grad,grad_opt_unroll);
% t_grad = real(cell2mat(grad.rho_grad));
 options = struct('GradObj','on','Display','iter','LargeScale','off','HessUpdate','lbfgs','InitialHessType','identity','GoalsExactAchieve',0);
 [opt_grad,cost] = fminlbfgs(main,grad_opt_unroll,options);
% %rollGrad=RollGradient(opt_grad,init_grad);
% for i = 1:7
%     params.rho(i,1) = {opt_grad(i)};
% end
% %params.rho(8,1) = par
% [ x_output2,net_forwarded2 ] = Forward( N,constants,params,net);
% k = reshape(cell2mat(x_output2),[size(img,1),size(img,1)]);
% imshow(real(k));
% hold on;
% imshow(real(img));
