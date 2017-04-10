% Authored by Rick~
clc;
clear all;
img = imread('IM-0001-0001.jpg');
img = rgb2gray(img);
img = img(256:258,256:258);

N = 10;
[constants,params,net]  = Init( img,N );
[ x_output,net_forwarded ] = Forward( N,constants,params,net);
[ E,grad ] = Back( img,constants,params,net);

unrollGrad=UnrollGradient(grad);
options = struct('GradObj','on','Display','iter','LargeScale','off','HessUpdate','lbfgs','InitialHessType','identity','GoalsExactAchieve',0);
[x2,fval2] = fminlbfgs(@myfun,x0,options);
rollGrad=RollGradient(unrollGrad,grad);