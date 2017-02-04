%
%  MultiplierUpdateLayerGradient.m
%  ADMM-NET
%
%  Created by Wang Han.SCU on 22/10/16.
%  Copyright (C) 2016 Deep ADMM NETWORK. SCU. All rights reserved.

function [E_2_eta_n,E_2_beta_nMinus1,E_2_c_n,E_2_z_n] = MultiplierUpdateLayerGradient(E_2_beta_n,c_n,z_n,eta_n,L)

    % This function is aimed to compute gradient in multiplier update layer
    % output:
    % E_2_eta_n is the gradient between cost E and eta(n)
    % E_2_beta_nMinus1 is the gradient between cost E and beta(n-1)
    % E_2_c_n is the gradient between cost E and c(n)
    % E_2_z_n is the gradient between cost E and z(n)
    
    % input:
    % E_2_beta_n is the gradient between cost E and beta(n)
    % beta_n is the value of n th Multiplier update Layer
    % c_n is the return value of n th Convolution Layer
    % z_n is the return value of n th Nonlinear Transform Layer
    % eta_n is the learnable parameter
    % L is the number of z in column, which has been definited before
    
    beta_n_2_eta_n = cell(1,L);
    beta_n_2_beta_nMinus1 = cell(1,L);
    beta_n_2_c_n = cell(1,L);
    beta_n_2_z_n = cell(1,L);
    
    E_2_eta_n = cell(1,L);
    E_2_beta_nMinus1 = cell(1,L);
    E_2_c_n = cell(1,L);
    E_2_z_n = cell(1,L);

    I_n = eye(size(c_n{1},1)); %I_n is an identity matrix sized N x N
    
    for l = 1:L
        beta_n_2_eta_n{l} = c_n{l} - z_n{l};
        beta_n_2_c_n{l} = eta_n{l};
        beta_n_2_z_n{l} = -eta_n{l};
        beta_n_2_beta_nMinus1{l} = I_n;
        E_2_eta_n{l} = beta_n_2_eta_n{l}'*E_2_beta_n{l};
        % these gradients will be used in upcoming steps:
        E_2_c_n{l} = beta_n_2_c_n{l}' *E_2_beta_n{l};
        E_2_z_n{l} = beta_n_2_z_n{l}' *E_2_beta_n{l};
        E_2_beta_nMinus1{l} = beta_n_2_beta_nMinus1{l}' * E_2_beta_n{l};
    end
end
