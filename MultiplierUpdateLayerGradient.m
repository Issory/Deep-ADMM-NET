%
%  MultiplierUpdateLayerGradient.m
%  ADMM-NET
%
%  Created by Wang Han.SCU on 22/10/16.
%  Copyright (C) 2016 Deep ADMM NETWORK. SCU. All rights reserved.

function [E2beta_gradient,E2eta_gradient] = MultiplierUpdateLayerGradient(...
        eta_n,c_n,z_n,L,x_next_gradient,x_next2beta_gradient,beta_next_gradient,...
        beta_next2beta_gradient,z_next_gradient,z_next2beta_gradient,is_N)

    % This function is aimed to compute gradient in multiplier update layer
    % output:
    % beta2eta_gradient is the gradient between beta(n) and eta(n)
    % beta2beta_gradient is the gradient between beta(n) and beta(n-1)
    % beta2c_gradient is the gradient between beta(n) and c(n)
    % beta2z_gradient is the gradient between beat(n) and z(n)
    % input:
    % eta_n is the learnable parameter
    % beta_n is the value of n th Multiplier update Layer
    % c_n is the return value of n th Convolution Layer
    % z_n is the return value of n th Nonlinear Transform Layer
    % L is the number of z in column, which has been definited before
    
    length = size(c_n,1);
    beta2eta_gradient = cell(length,1);
    beta2beta_gradient = cell(length,1);
    beta2c_gradient = cell(length,1);
    beta2z_gradient = cell(length,1);
    E2beta_gradient = cell(length,1);
    E2eta_gradient = cell(length,1);
    I_n = {eye(length)}; %I_n is an identity matrix sized N x N
    for l = 1:L
        beta2eta_gradient{l} = c_n{l} - z_n{l};
        beta2beta_gradient{l} = I_n;
        beta2c_gradient{l} = eta_n{l} * I_n;
        beta2z_gradient{l} = -eta_n{l} * I_n;
        if (is_N)
            E2beta_gradient{l} = x_next_gradient{l} * x_next2beta_gradient{l};
        else
            E2beta_gradient{l} = beta_next_gradient{l} * beta_next2beta_gradient{l} +...
                z_next_gradient{l} * z_next2beta_gradient{l} + x_next_gradient{l} * x_next2beta_gradient{l};
        end
        E2eta_gradient{l} = beta2eta_gradient{l}'*E2beta_gradient{l};
        E2c_gradient{l} = beta2c_gradient{l}' *E2beta_gradient{l};
        E2z_gradient{l} = beta2z_gradient{l}' *E2beta_gradient{l};
        
    end
 
% these gradients will be used in upcoming steps:
E_2_c2_first = beta2_2_c2' * E_2_beta2;
E_2_z2_second = beta2_2_z2' * E_2_beta2;
E_2_beta1_first = beta2_2_beta1' * E_2_beta2;
end
