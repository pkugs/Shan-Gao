clear
%% transfer function of microphone array and HOA
dataRootPath = 'gaoData/'
hoa_tf_file=[dataRootPath, 'signal/train/HOA-1024-r042cm-48khz-with-2000p.mat'];
mic_tf_file=[dataRootPath, 'signal/train/TF-1024-r042cm-48khz-with-2000p.mat'];
load(hoa_tf_file);
hoa_tf=data;
load(mic_tf_file);
mic_tf=tf;
% mic_tf=data;
%% frame number and frame len
sig_num=20;
frame_len=1024;
%% output folder
mic_out_folder=[dataRootPath, 'signal/train/uniform_mic_noisy30dB/mic_sig/'];
hoa_out_folder=[dataRootPath, 'signal/train/uniform_mic_noisy30dB/hoa_sig/'];

if ~exist(mic_out_folder,'dir')
    mkdir(mic_out_folder)
end
if ~exist(hoa_out_folder,'dir')
    mkdir(hoa_out_folder)
end
pos_len=1950;
file_name_offset = 0
next_posi_offset = 0;
size(hoa_tf)
% final_out_mic=zeros(2000,sig_num,frame_len*32);
% final_out_hoa=zeros(2000,sig_num,frame_len*25);
%% source signal 
signal=randn(1024, sig_num);
fft_signal=fft(signal);
% fft_signal=fft_signal./abs(fft_signal)*30;
for angle_ii=1: 1950
    angle_ii
    %% source one
    t_hoa_tf=hoa_tf{angle_ii};
    t_mic_tf=mic_tf{angle_ii};
    fft_hoa=fft(t_hoa_tf);
    fft_mic=fft(t_mic_tf);
%     signal=randn(1024,sig_num);
%     fft_signal=fft(signal);
%     fft_signal=fft_signal./abs(fft_signal)*30;
    %% source two
    next_pos=ceil(rand(1)*pos_len) + next_posi_offset;
    fft_hoa_p=fft(hoa_tf{next_pos});
    fft_mic_p=fft(mic_tf{next_pos});
    signal_p=randn(1024, sig_num);
    fft_signal_p=fft(signal_p);
%     fft_signal_p=fft_signal_p./abs(fft_signal_p)*30;
    %% source three
    next_pos2=ceil(rand(1)*pos_len) + next_posi_offset;
    fft_hoa_p2=fft(hoa_tf{next_pos2});
    fft_mic_p2=fft(mic_tf{next_pos2});
    signal_p2=randn(1024,sig_num);
    fft_signal_p2=fft(signal_p2);
%     fft_signal_p2=fft_signal_p2./abs(fft_signal_p2)*30;
    %% source four
    next_pos3=ceil(rand(1)*pos_len) + next_posi_offset;
    fft_hoa_p3=fft(hoa_tf{next_pos3});
    fft_mic_p3=fft(mic_tf{next_pos3});
    signal_p3=randn(1024,sig_num);
    fft_signal_p3=fft(signal_p3);
%     fft_signal_p3=fft_signal_p3./abs(fft_signal_p3)*30;
    %% source five
    next_pos4=ceil(rand(1)*pos_len) + next_posi_offset;
    fft_hoa_p4=fft(hoa_tf{next_pos4});
    fft_mic_p4=fft(mic_tf{next_pos4});
    signal_p4=randn(1024,sig_num);
    fft_signal_p4=fft(signal_p4);
    

    out_mic=zeros(sig_num,frame_len*32);
    out_hoa=zeros(sig_num,frame_len*25);
    for sig_ii=1:sig_num
        temp_sig=fft_signal(:,sig_ii);
%         temp_sig=mean(fft_signal,2);
        temp_sig_mic=repmat(temp_sig,1,32);
        temp_sig_hoa=repmat(temp_sig,1,25);
        rec_mic=fft_mic.*temp_sig_mic;
        rec_hoa=fft_hoa.*temp_sig_hoa;
%         rec_mic=fft_mic*40;
%         rec_hoa=fft_hoa*40;
        
        temp_sig_p=fft_signal_p(:,sig_ii);
        temp_sig_mic_p=repmat(temp_sig_p,1,32);
        temp_sig_hoa_p=repmat(temp_sig_p,1,25);
        rec_mic_p=fft_mic_p.*temp_sig_mic_p;
        rec_hoa_p=fft_hoa_p.*temp_sig_hoa_p;
%         rec_mic_p=fft_mic_p*40;
%         rec_hoa_p=fft_hoa_p*40;

        
        temp_sig_p2=fft_signal_p2(:,sig_ii);
        temp_sig_mic_p2=repmat(temp_sig_p2,1,32);
        temp_sig_hoa_p2=repmat(temp_sig_p2,1,25);
        rec_mic_p2=fft_mic_p2.*temp_sig_mic_p2;
        rec_hoa_p2=fft_hoa_p2.*temp_sig_hoa_p2;
%         rec_mic_p2=fft_mic_p2*40;
%         rec_hoa_p2=fft_hoa_p2*40;

        
        temp_sig_p3=fft_signal_p3(:,sig_ii);
        temp_sig_mic_p3=repmat(temp_sig_p3,1,32);
        temp_sig_hoa_p3=repmat(temp_sig_p3,1,25);
        rec_mic_p3=fft_mic_p3.*temp_sig_mic_p3;
        rec_hoa_p3=fft_hoa_p3.*temp_sig_hoa_p3;
%         rec_mic_p3=fft_mic_p3*40;
%         rec_hoa_p3=fft_hoa_p3*40;
        temp_sig_p4=fft_signal_p4(:,sig_ii);
        temp_sig_mic_p4=repmat(temp_sig_p4,1,32);
        temp_sig_hoa_p4=repmat(temp_sig_p4,1,25);
        rec_mic_p4=fft_mic_p4.*temp_sig_mic_p4;
        rec_hoa_p4=fft_hoa_p4.*temp_sig_hoa_p4;

        
       %% add noise to signal
        % rec_mic=awgn(rec_mic,30,'measured');
        % rec_mic_p=awgn(rec_mic_p,30,'measured');
        % rec_mic_p2=awgn(rec_mic_p2,30,'measured');
        % rec_mic_p3=awgn(rec_mic_p3,30,'measured');
        % rec_mic_p4=awgn(rec_mic_p4,30,'measured');
        %% sum the signals from different sources
        out_mic_fifth(sig_ii,:)=sig_reshape(rec_mic)+sig_reshape(rec_mic_p)...
            +sig_reshape(rec_mic_p2)+sig_reshape(rec_mic_p3)+sig_reshape(rec_mic_p4);
        out_hoa_fifth(sig_ii,:)=sig_reshape(rec_hoa)+sig_reshape(rec_hoa_p)...
            +sig_reshape(rec_hoa_p2)+sig_reshape(rec_hoa_p3)+sig_reshape(rec_hoa_p4);
        out_mic_fourth(sig_ii,:)=sig_reshape(rec_mic)+sig_reshape(rec_mic_p)...
            +sig_reshape(rec_mic_p2)+sig_reshape(rec_mic_p3);
        out_hoa_fourth(sig_ii,:)=sig_reshape(rec_hoa)+sig_reshape(rec_hoa_p)...
            +sig_reshape(rec_hoa_p2)+sig_reshape(rec_hoa_p3);
        out_mic_trip(sig_ii,:)=sig_reshape(rec_mic)+sig_reshape(rec_mic_p)+sig_reshape(rec_mic_p2);
        out_hoa_trip(sig_ii,:)=sig_reshape(rec_hoa)+sig_reshape(rec_hoa_p)+sig_reshape(rec_hoa_p2);
        out_mic_double(sig_ii,:)=sig_reshape(rec_mic)+sig_reshape(rec_mic_p);
        out_hoa_double(sig_ii,:)=sig_reshape(rec_hoa)+sig_reshape(rec_hoa_p);
        out_mic(sig_ii,:)=sig_reshape(rec_mic);
        out_hoa(sig_ii,:)=sig_reshape(rec_hoa);
    end
%     final_out_mic(angle_ii,:,:)=out_mic;
%     final_out_hoa(angle_ii,:,:)=out_hoa;
    hoa_file=[hoa_out_folder,num2str(angle_ii+0000+file_name_offset)];
    mic_file=[mic_out_folder,num2str(angle_ii+0000+file_name_offset)]; 
    hoa_file_double=[hoa_out_folder,num2str(angle_ii+2000+file_name_offset)];
    mic_file_double=[mic_out_folder,num2str(angle_ii+2000+file_name_offset)]; 
    hoa_file_trip=[hoa_out_folder,num2str(angle_ii+4000+file_name_offset)];
    mic_file_trip=[mic_out_folder,num2str(angle_ii+4000+file_name_offset)]; 
    hoa_file_four=[hoa_out_folder,num2str(angle_ii+6000+file_name_offset)];
    mic_file_four=[mic_out_folder,num2str(angle_ii+6000+file_name_offset)];
    hoa_file_five=[hoa_out_folder,num2str(angle_ii+8000+file_name_offset)];
    mic_file_five=[mic_out_folder,num2str(angle_ii+8000+file_name_offset)]; 
    data=out_mic;
    save(mic_file,'data')
    data=out_hoa;
    save(hoa_file,'data')
    data=out_mic_double;
    save(mic_file_double,'data')
    data=out_hoa_double;
    save(hoa_file_double,'data')
    data=out_mic_trip;
    save(mic_file_trip,'data')
    data=out_hoa_trip;
    save(hoa_file_trip,'data')
    data=out_mic_fourth;
    save(mic_file_four,'data')
    data=out_hoa_fourth;
    save(hoa_file_four,'data')
%     data=out_mic_fifth;
%     save(mic_file_five,'data')
%     data=out_hoa_fifth;
%     save(hoa_file_five,'data')
end
% hoa_file=[hoa_out_folder,'guass-hoa-tf-10f.mat'];
% mic_file=[mic_out_folder,'guass-mic-tf-10f.mat'];   
% data=final_out_mic;
% save(mic_file,'data','-v7.3')
% data=final_out_hoa;
% save(hoa_file,'data','-v7.3')

function out=sig_reshape(signal)
sig=signal.';
[ch_num,sig_len]=size(sig);
sig=sig(:,1:sig_len/2);
len_sig=reshape(sig,[1,ch_num*sig_len/2]);
cp_sig=[real(len_sig);imag(len_sig)];
out=reshape(cp_sig,[1,ch_num*sig_len]);
end
        
    