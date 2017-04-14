function output = sft_func_bp_z2beta(L,sum,q)

%  ADMM-NET
%
%  Created by Liao Tinghui.SCU on 27/12/16.
%  Copyright (C) 2016 Deep ADMM NETWORK. SCU. All rights reserved.

% sum  = c_n+beta_(n-1)
% L is the number of x
output = zeros(1,L);
step = 2/(L-1);
L_q = size(q,2);
p = zeros(1,L_q);
for i = 1:L_q
  p(i) = -1+(i-1)*step; %get the x axis
end
length = size(sum,1);%Image vector size
for j = 1:length
for i = 1:length
    if sum(j)<=p(1) || sum(j)>=p(L_q)
        output(j,i)=1;
    else
        %inx = find(sum(j),q);
        inx = real(floor((sum(j)-p(1))/(p(2)-p(1))));
        output(j,i) = 1-(q(inx+1)-q(inx))/(p(2)-p(1));
    end
end
end



end
    function position = find(number,vec)
        position = 1;
        for inx = 1:(size(vec,2)-1)
            if number<=vec(inx+1)
                position = inx;
                break;
            end
        end
    end