%
%  RollGradinet.m
%  ADMM-NET
%
%  Created by Wang Han.SCU on 8/4/17.
%  Copyright (C) 2016 Deep ADMM NETWORK. SCU. All rights reserved.

function roll_grad = RollGradinet(unroll_grad,grad)
% This function is aim to unroll the gradient struct 
% 
% input:
% unroll_grad is a vector which is unrolled gradient
% output: 
% roll_grad is a struct which is rolled gradient 
%
roll_grad=struct;

index =1;


% roll_grad.x_grad
unroll_x_grad=cell(size(grad.x_grad,1),1);
for i=1:size(grad.x_grad,1)
        length=size(grad.x_grad{i},1);
        unroll_x_grad{i}=unroll_grad(index:index+length-1);
        index=index+length;
end
roll_grad.x_grad=unroll_x_grad;

% roll_grad.z_grad
unroll_z_grad=cell(size(grad.z_grad,1),1);
for i=1:size(grad.z_grad,1)
        length=size(grad.z_grad{i},1);
        unroll_z_grad{i}=unroll_grad(index:index+length-1);
        index=index+length;
end
roll_grad.z_grad=unroll_z_grad;

% roll_grad.beta_grad
unroll_beta_grad=cell(size(grad.beta_grad,1),1);
for i=1:size(grad.beta_grad,1)
        length=size(grad.beta_grad{i},1);
        unroll_beta_grad{i}=unroll_grad(index:index+length-1);
        index=index+length;
end
roll_grad.beta_grad=unroll_beta_grad;

% roll_grad.c_grad
unroll_c_grad=cell(size(grad.c_grad,1),1);
for i=1:size(grad.c_grad,1)
        length=size(grad.c_grad{i},1);
        unroll_c_grad{i}=unroll_grad(index:index+length-1);
        index=index+length;
end
roll_grad.c_grad=unroll_c_grad;



end
