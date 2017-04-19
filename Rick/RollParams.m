
function roll_params = RollParams(unroll_params,params)
% This function is aim to unroll the gradient struct 
% 
% input:
% unroll_grad is a vector which is unrolled gradient
% output: 
% roll_grad is a struct which is rolled gradient 
%
roll_params=struct;

index =1;
% roll_grad.gamma_grad
unroll_gamma=cell(size(params.gamma,1),1);
for i=1:size(params.gamma,1)
        length=size(params.gamma{i},1);
        unroll_gamma{i}=unroll_params(index:index+length-1);
        index=index+length;
end
roll_params.gamma=unroll_gamma;

% roll_grad.eta_grad
unroll_eta=cell(size(params.eta,1),1);
for i=1:size(params.eta,1)
        length=size(params.eta{i},1);
        unroll_eta{i}=unroll_params(index:index+length-1);
        index=index+length;
end
roll_params.eta=unroll_eta;

% roll_grad.rho_grad
unroll_rho=cell(size(params.rho,1),1);
for i=1:size(params.rho_grad,1)
        length=size(params.rho{i},1);
        unroll_rho{i}=unroll_params(index:index+length-1);
        index=index+length;
end
roll_params.rho=unroll_rho;

% roll_grad.q_grad
unroll_q=cell(size(params.q,1),1);
for i=1:size(params.q,1)
        length=size(params.q{i},1);
        unroll_q{i}=unroll_params(index:index+length-1);
        index=index+length;
end
roll_params.q=unroll_q;

% roll_grad.w_grad
unroll_w=cell(size(params.w,1),1);
for i=1:size(params.w_grad,1)
        length=size(params.w{i},1);
        unroll_w{i}=unroll_params(index:index+length-1);
        index=index+length;
end
roll_params.w=unroll_w;





end
