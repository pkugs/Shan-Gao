clear
clc
dist=20;
fs=48000;
tfLen=1024;
source_list=(0:5:355)/180*pi;
tf=cell(72,1);
singleTf=zeros(tfLen,32);
for source_ii=1:length(source_list)
    source_ii
    source_angle=source_list(source_ii);
    temp_angle=jiaodu2(source_angle,0);
    for point_ii=1:32
        include_angle=temp_angle(point_ii,3);
        singleTf(:,point_ii) = sphere_hrtf(fs,tfLen,include_angle,dist);
    end
    tf{source_ii}=singleTf;
end
save('TF-1024-r042cm-48khz-0-355.mat','tf')
