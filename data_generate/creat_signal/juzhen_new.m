function y=juzhen_new(order)          %????????

% x=jiaodu();
x=mic_angle();
% load('/data/gs/code box/microphone distribution/guass_32p.mat');
% load('mic_coordinate')
% [x(:,1),x(:,2),x(:,3)]=cart2sph(mic_coor(:,4),mic_coor(:,3),mic_coor(:,2));
mic_number=length(x(:,1));
y=zeros(mic_number,(order+1)^2);
for mic_ii=1:mic_number
    y(mic_ii,:)=getSH(order,[x(mic_ii,1),pi/2-x(mic_ii,2)],'real');
end
end