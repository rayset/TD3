clear all
close all
work={'workspace3.mat','workspace4.mat'};
figure()
hold on
for kij=1:2
    load(work{kij})
    s_plot_reduced=s_plot_reduced(size(s_plot_reduced,1)-50:end,:);
    s_bar_p(kij,1)=sum(sum(s_plot_reduced<-0.1));
    s_bar_p(kij,2)=sum(sum(s_plot_reduced<0))- s_bar(1)   ;
    s_bar_p(kij,3)=sum(sum(s_plot_reduced<0.1)) - s_bar(1) - s_bar(2) -sum(sum(s_plot_reduced==0));
    s_bar_p(kij,4)=sum(sum(s_plot_reduced>0.1));
    s_bar_p(kij,:)=s_bar_p(kij,:)/sum(s_bar_p(kij,:));
end

b=bar(s_bar_p'*100)
ylabel('Percentage of time [%]')
xticks([1,2,3,4])
xticklabels({'x<-0.1','-0.1 \leq x < 0',' 0 \leq x < 0.1',' x > 0.1'})
x0=100;
y0=100;
width=550;
height=250;
set(gcf,'position',[x0,y0,width,height])
ax = gca;
ax.XGrid = 'off';
ax.YGrid = 'on';
