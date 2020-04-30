close all
figure()
clear avreaged_dr
clear avreaged_Q
clear d_std
clear Q_std
clear error
clear s_bar
clear s_plot
subplot(1,2,1)
plot([1:size(d_rewards,2)],d_rewards,[1:size(d_rewards,2)],Q_values')
legend('discounted rewards','Q0')
subplot(1,2,2)
for i=1:size(d_rewards,2)/50
avreaged_dr(i)=mean(d_rewards(1+(i-1)*50:(i-1)*50+50));
avreaged_Q(i)=mean(Q_values(1+(i-1)*50:(i-1)*50+50));
end
plot([1:size(d_rewards,2)/50],avreaged_dr,[1:size(d_rewards,2)/50],avreaged_Q)
legend('discounted rewards','Q0')

figure()
hold on
for i=1:size(d_rewards,2)/50
d_std(i,:)=d_rewards(1+(i-1)*50:(i-1)*50+50);
Q_std(i,:)=Q_values(1+(i-1)*50:(i-1)*50+50);
error(i,:)=d_rewards(1+(i-1)*50:(i-1)*50+50)'-Q_values(1+(i-1)*50:(i-1)*50+50);
end

a=stdshade(d_std',0.2)
b=stdshade(Q_std',0.2,'blue')
c=stdshade(error',0.2,'green')
grid()
xlim([1,size(Q_std,1)])
ylim([-10,25])
xticks([1,6,11,16,21,26,31,36,41,46,51])
xticklabels({0,5,10,15,20,25,30,40,50})
legend([a,b,c],'discounted reward','Q0','error')
xlabel('Validation N°')
x0=100;
y0=100;
width=550;
height=250;
set(gcf,'position',[x0,y0,width,height])


%
%
%
j=0;
k=1;
for i=1:size(states,2)
    if sum(states{i})==0
        j=j+1;
        i=i+1;
        k=1;
    end
    s_plot(j,k)=states{i}(1);
    k=k+1;
end
s_plot_reduced=s_plot(size(s_plot,1)-50:end,:)

s_bar(1)=sum(sum(s_plot_reduced<-0.1))
s_bar(2)=sum(sum(s_plot_reduced<0))- s_bar(1)   
s_bar(3)=sum(sum(s_plot_reduced<0.1)) - s_bar(1) - s_bar(2) -sum(sum(s_plot_reduced==0))
s_bar(4)=sum(sum(s_plot_reduced>0.1))

figure
plot(s_plot_reduced')
figure
bar(s_bar/sum(s_bar)*100)
ylabel('Percentage of time [%]')
xticklabels({'x<-0.1','-0.1 \leq x < 0',' 0 \leq x < 0.1',' x > 0.1'})
x0=100;
y0=100;
width=550;
height=250;
set(gcf,'position',[x0,y0,width,height])

figure()
stdshade(s_plot_reduced,0.2,[0.6350 0.0780 0.1840])
xlim([1,size(s_plot_reduced,2)])
xticks([1,26,51,76,101,126,151])
xticklabels({0,25,50,75,100,125,150})
xlabel('Time (s)')
ylabel('x [m]')
ylim([-0.2,1.100001])
grid()
x0=100;
y0=100;
width=550;
height=250;
set(gcf,'position',[x0,y0,width,height])

