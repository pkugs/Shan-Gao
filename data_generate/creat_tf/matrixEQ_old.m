% matrix for equalization, composed of 1/W_m(kR) and regularization factor
% F_m(kR) (based on the Daniel 2006 paper)
function EQ = matrixEQ_old(N,f,r,speed)
% N--highest order(starting from 0)   f--frequency of sound in Hz
% r--distance from the point to the center
c = speed;   % speed of sound
k = 2*pi*f/c;
kr = k*r;
columns = zeros(1,(N+1)^2);
function x = EQn(n,lambda)
% bn = 1i*sqrt(pi/(2*kr))*(besselj(n+1/2,kr)*bessely(n+3/2,kr)-besselj(n+3/2,kr)*bessely(n+1/2,kr))/ ...
% (n/kr*besselh(n+1/2,2,kr)-besselh(n+3/2,2,kr));
temp = (kr)^2*sqrt(pi/(2*kr))*(n/kr*besselh(n+1/2,2,kr) - besselh(n+3/2,2,kr));
bn = 1i/temp;
%bn = 1/(kr)^2/(sqrt(pi/(2*kr))*(n/kr*besselh(n+1/2,2,kr)-besselh(n+3/2,2,kr)));
Wn = (1j)^n*bn;      %have questions (solved)
% Wn = bn;
Fn = (abs(Wn))^2/((abs(Wn))^2+lambda^2);

x = 1/Wn*Fn;
% x=1/Wn;

end
lambda = 3e-3;
%     function lambda = lambdavalue(n)
%         switch(n)
%             case 0
%                 lambda = (f<=40)*5e8;
%             case 1
%                 lambda = 2.2e-2;
%             case 2
%                 lambda = (f<=4145)*8.6e-3;
%             case 3
%                 lambda = 8.6e-3;
%             case 4
%                 lambda = (f<=15000)*7e-3;
%         end
%     end
% 
% for j = 0:N
%     columns(:,(j^2+1):(j+1)^2) = EQn(j,lambdavalue(j));   % value of lambda not determined
% end
for j = 0:N
    columns(:,(j^2+1):(j+1)^2) = EQn(j,lambda);   % value of lambda not determined
end
EQ = diag(columns);

end







