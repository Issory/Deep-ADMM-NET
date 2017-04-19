%
%  UnrollGradinet.m
%  ADMM-NET
%
%  Created by Wang Han.SCU on 8/4/17.
%  Copyright (C) 2016 Deep ADMM NETWORK. SCU. All rights reserved.

function unroll_grad = UnrollGradient(roll_grad)
% This function is aim to unroll the gradient struct 
% 
% input: 
% roll_grad is a struct which is rolled gradient 
% ouput:
% unroll_grad is a vector which is unrolled gradient
%

% roll_grad.x_grad
unroll_x_grad=[];
for i=1:size(roll_grad.x_grad,1)
        unroll_x_grad=[unroll_x_grad;roll_grad.x_grad{i}(:)];
end

% roll_grad.z_grad
unroll_z_grad=[];
for i=1:size(roll_grad.z_grad,1)
        unroll_z_grad=[unroll_z_grad;roll_grad.z_grad{i}(:)];
end

% roll_grad.beta_grad
unroll_beta_grad=[];
for i=1:size(roll_grad.beta_grad,1)
        unroll_beta_grad=[unroll_beta_grad;roll_grad.beta_grad{i}(:)];
end

% roll_grad.c_grad
unroll_c_grad=[];
for i=1:size(roll_grad.c_grad,1)
        unroll_c_grad=[unroll_c_grad;roll_grad.c_grad{i}(:)];
end



unroll_grad=[unroll_x_grad;unroll_z_grad;unroll_beta_grad;unroll_c_grad];%unroll_gamma_beta];
end

