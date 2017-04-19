function [ E,grad_opt_unroll ] = main_process(img,N,constants,params,net,init_grad,grad_opt_unroll)

[ x_output,net_forwarded ] = Forward( N,constants,params,net);
[ E,grad_opt_unroll ] = Back_opt(grad_opt_unroll,img,constants,params,init_grad,net_forwarded);
end