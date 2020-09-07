function [azi,ele]=local_func(order,hoa)

% order=4;
% hoa=getSH(4,[0.87,pi/2+0.36],'real');

azi_list=0:3*pi/180:2*pi;
ele_list=pi/2:-3*pi/180:-pi/2;

frame_image=zeros(length(azi_list),length(ele_list));
for azi_ii=1:length(azi_list)
    for ele_ii=1:length(ele_list)
        frame_image(azi_ii,ele_ii)=getSH(order,[azi_list(azi_ii),pi/2-ele_list(ele_ii)],'real')*hoa.';
    end
end
frame_image=abs(frame_image);
locs=find(imregionalmax(frame_image));
peak_vals=frame_image(locs);
[~,max_peak]=max(peak_vals);
locs=locs(max_peak);
[azi_pos,ele_pos]=ind2sub(size(frame_image),locs);
azi=azi_list(azi_pos);
ele=ele_list(ele_pos);

end