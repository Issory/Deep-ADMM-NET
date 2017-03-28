%
%  NonlinearTransformLayerGradient.m
%  ADMM-NET
%
%  Created by Wang Han.SCU on 22/10/16.
%  Copyright (C) 2016 Deep ADMM NETWORK. SCU. All rights reserved.

function [E_2_q_n,E_2_beta_nMinus1,E_2_c_n] = NonlinearTransformLayerGradient(E_2_z_n,c_n,beta_nMinus1,q,L)
    % This function is aimed to compute the gradient of nonlinera thansform layer
    % output:
    % E_2_q_n is the gradient between E and q(n)
    % E_2_beta_nMinus1 is the gradient between E and beta(n-1)
    % E_2_c_n is the gradient between E and c(n)
    % input:
    % E_2_z_n is the gradient between E and z(n)
    % c_n is the value of n th convolution layer
    % beta_nMinus1 is the value of n-1 th Multiplier update Layer
    % p is the number of interval that belongs to [-1,1]
    % q is a Nc*L matrix deciding the values in y axis of PLF
    % L is the number of z in column, which has been definited before.
    
    z_n_2_q_n = {zeros(1,L)};
    z_n_2_beta_nMinus1 = {zeros(1,L)};
    z_n_2_c_n = {zeros(1,L)};
    
    E_2_q_n = {zeros(1,L)};
    E_2_beta_nMinus1 = {zeros(1,L)};
    E_2_c_n = {zeros(1,L)};
    
    % tips: maybe have some problems
    for l = 1:L
        z_n_2_q_n{l}= sft_func_bp_z2q(size(q{l},2),c_n{1}+beta_nMinus1{1});
        z_n_2_beta_nMinus1{l} = sft_func_bp_z2beta(size(c_n{l},1),c_n{1}+beta_nMinus1{1},q{l});
        z_n_2_c_n{l} = z_n_2_beta_nMinus1{l};

        % these gradients can be used directly
        E_2_q_n{l} = z_n_2_q_n{l}' * E_2_z_n{l};
        % these gradients will be used in upcoming steps:
        E_2_beta_nMinus1{l} = z_n_2_beta_nMinus1{l}' * E_2_z_n{l};
        E_2_c_n{l} = z_n_2_c_n{l}' * E_2_z_n{l};
    end
end