clear
% dataset path
dataRootPath = 'gaoData/'
load('2000p.mat');
% data(:,1)=atan2(rn(:,2),rn(:,1));
% data(:,2)=atan(rn(:,3)./sqrt(rn(:,1).^2+rn(:,2).^2));
% azi_list=(0:6:360)/180*pi;
% ele_list=(90:-5:-90)/180*pi;
ele_list=0;
hoa_tf=[];
count = 1;
length(rn(:,1))
for ii = 1: length(rn(:,1))
    ii
    [azim, elev, r]=cart2sph(rn(ii,1),rn(ii,2),rn(ii,3));
    t_hoa=zeros(1024,25);
    
    %% the amplitude of the HOA transfer function is determined by the 
    %  decomposition result of the microphone array tf, here is set to 0.06
    h_list=getSH(4,[azim, pi/2-elev],'real');
    h_list=h_list/h_list(1)*0.06;
    %% time delay of determined by the decomposition result of the microphone array tf,
    % here is set to 465, corresponding to the distance of 20m
    t_hoa(465,:)=h_list;
    hoa_tf{count}=t_hoa;
    count = count + 1
end
data=hoa_tf;
save([dataRootPath, 'signal/train/HOA-1024-r042cm-48khz-with-2000p.mat'], 'data')