% clear
% clc
% hoa_tf_file='/data/gs/mic_to_hoa_new/signal/train/HOA-1024-r042cm-48khz-with-2000p.mat';
% mic_tf_file='/data/gs/mic_to_hoa_new/signal/train/TF-1024-r042cm-48khz-with-2000p.mat';
% load(hoa_tf_file);
% hoa_tf=data;
% load(mic_tf_file);
% mic_tf=data;
% fft_hoa=zeros(2000,1024,25);
% fft_mic=zeros(2000,1024,32);
% for angle_ii=1:2000
%     s_hoa=hoa_tf{angle_ii};
%     s_mic=mic_tf{angle_ii};
%     fft_hoa(angle_ii,:,:)=fft(s_hoa);
%     fft_mic(angle_ii,:,:)=fft(s_mic);
% end

for f_ii=2:513
    t_hoa=fft_hoa(:,f_ii,:);
    t_mic=fft_mic(:,f_ii,:);
    t_hoa=squeeze(t_hoa);
    t_mic=squeeze(t_mic);
    t_hoa=t_hoa.';
    t_mic=t_mic.';
%     inv_mic=(t_mic.'*t_mic)^(-1)*t_mic.';
    inv_mic=pinv(t_mic);
    t=t_mic*inv_mic;
    
    T{f_ii-1}=t_hoa*inv_mic;
end
save('cal_T.mat','T')
