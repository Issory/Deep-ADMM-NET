
function unroll_grad = UnrollGradient_opt(roll_grad)
% This function is aim to unroll the gradient struct 
% 
% input: 
% roll_grad is a struct which is rolled gradient 
% ouput:
% unroll_grad is a vector which is unrolled gradient
%

% roll_grad.gamma_grad
unroll_gamma_grad=[];
for i=1:size(roll_grad.gamma_grad,1)
        unroll_gamma_grad=[unroll_gamma_grad;roll_grad.gamma_grad{i}(:)];
end

% roll_grad.eta_grad
unroll_eta_grad=[];
for i=1:size(roll_grad.eta_grad,1)
        unroll_eta_grad=[unroll_eta_grad;roll_grad.eta_grad{i}(:)];
end

% roll_grad.rho_grad
unroll_rho_grad=[];
for i=1:size(roll_grad.rho_grad,1)
        unroll_rho_grad=[unroll_rho_grad;roll_grad.rho_grad{i}(:)];
end

% roll_grad.q_grad
unroll_q_grad=[];
for i=1:size(roll_grad.q_grad,1)
        unroll_q_grad=[unroll_q_grad;roll_grad.q_grad{i}(:)];
end

% roll_grad.w_grad
unroll_w_grad=[];
for i=1:size(roll_grad.w_grad,1)
        unroll_w_grad=[unroll_w_grad;roll_grad.w_grad{i}(:)];
end





unroll_grad=[unroll_gamma_grad;unroll_eta_grad;unroll_rho_grad;unroll_q_grad;unroll_w_grad];%unroll_gamma_beta];
end

