
function roll_grad = RollGradient_opt(unroll_grad,grad)
% This function is aim to unroll the gradient struct 
% 
% input:
% unroll_grad is a vector which is unrolled gradient
% output: 
% roll_grad is a struct which is rolled gradient 
%
roll_grad=struct;

index =1;
% roll_grad.gamma_grad
unroll_gamma_grad=cell(size(grad.gamma_grad,1),1);
for i=1:size(grad.gamma_grad,1)
        length=size(grad.gamma_grad{i},1);
        unroll_gamma_grad{i}=unroll_grad(index:index+length-1);
        index=index+length;
end
roll_grad.gamma_grad=unroll_gamma_grad;

% roll_grad.eta_grad
unroll_eta_grad=cell(size(grad.eta_grad,1),1);
for i=1:size(grad.eta_grad,1)
        length=size(grad.eta_grad{i},1);
        unroll_eta_grad{i}=unroll_grad(index:index+length-1);
        index=index+length;
end
roll_grad.eta_grad=unroll_eta_grad;

% roll_grad.rho_grad
unroll_rho_grad=cell(size(grad.rho_grad,1),1);
for i=1:size(grad.rho_grad,1)
        length=size(grad.rho_grad{i},1);
        unroll_rho_grad{i}=unroll_grad(index:index+length-1);
        index=index+length;
end
roll_grad.rho_grad=unroll_rho_grad;

% roll_grad.q_grad
unroll_q_grad=cell(size(grad.q_grad,1),1);
for i=1:size(grad.q_grad,1)
        length=size(grad.q_grad{i},1);
        unroll_q_grad{i}=unroll_grad(index:index+length-1);
        index=index+length;
end
roll_grad.q_grad=unroll_q_grad;

% roll_grad.w_grad
unroll_w_grad=cell(size(grad.w_grad,1),1);
for i=1:size(grad.w_grad,1)
        length=size(grad.w_grad{i},1);
        unroll_w_grad{i}=unroll_grad(index:index+length-1);
        index=index+length;
end
roll_grad.w_grad=unroll_w_grad;





end
