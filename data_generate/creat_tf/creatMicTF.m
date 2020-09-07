clear
clc
dist=20; % distance of the source
fs=48000;
tfLen=1024;  % transfer function length

% dataset path
dataRootPath = 'gaoData/'
load('2000p.mat'); % source distribution 
source_list=rn*dist;  

singleTf=zeros(tfLen,32);
x=mic_angle; % microphones distribution
rh=0.042; % array radius
for point_ii=1:32
    x2(point_ii,1)=cos(x(point_ii,1))*rh*cos(x(point_ii,2));
    x2(point_ii,2)=sin(x(point_ii,1))*rh*cos(x(point_ii,2));
    x2(point_ii,3)=rh*sin(x(point_ii,2));
end

%% add the position noise 
% noise=randn(32,3)*0.004;
% x2=x2+noise;
% load('/data/gs/code box/microphone distribution/test_guass_32p.mat');

%% creat tf
for s_ii=1:length(source_list(:,1))
    s_ii
    s_c=source_list(s_ii,:);
    for point_ii=1:32
        include_angle=acos((s_c(1)*x2(point_ii,1)+s_c(2)*x2(point_ii,2)...
            +s_c(3)*x2(point_ii,3))/(dist*rh));
        singleTf(:,point_ii) = sphere_hrtf(fs,tfLen,include_angle,dist);
    end    
    tf{s_ii}=singleTf;
end
save([dataRootPath, 'signal/train/TF-1024-r042cm-48khz-with-2000p.mat'],'tf')
