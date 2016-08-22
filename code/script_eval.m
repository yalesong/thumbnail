clear all; clc;

labels_dir = '../labels/';
results_dir = '../results/';
dataset_tsv = '../videos/yahoo_thumbnail_cikm2016.tsv';

% We use export_fig to export figures to eps format files. 
% Get it here: https://github.com/altmany/export_fig
export_fig_path = '__PATH_TO_EXPORT_FIG__';

% SIFTflow ditance threshold
theta = 0:0.0001:0.005;

% size of ranked list
topK = 5; 

% Model names, to be used in results
model_names={'Random','K-means Centroid','K-means Stillness','G.-LASSO',...
    'Beauty Rank','CNN','Ours Supervised','Ours Unsupervised'};

% Model directory names
models = {'random','kmeans_centroid','kmeans_stillness','glasso',...
    'beauty_rank','cnn','ours_supervised','ours_unsupervised'};
    
for i=1:numel(models),
    pred_dir{i} = [results_dir '/' models{i} '/'];
end

% Get list of video IDs
videos = dir([labels_dir '/*.txt']);
videos = cellfun(@(x) strrep(x,'.txt',''), {videos.name}, 'UniformOutput',false);

% Get the video categories
categories = cell(size(videos));
fid = fopen(dataset_tsv);
tline = fgetl(fid);
while ischar(tline)
    tok = textscan(tline,'%s','delimiter','\t');
    cat = strrep(strrep(tok{1}{2},' ',''),'&','');
    categories{find(strcmp(videos,tok{1}{1}))} = cat;
    tline = fgetl(fid);
end
fclose(fid);

%% Evaluate Precision@k
precision_k = zeros(numel(videos),numel(models),numel(theta),topK);
for i=1:numel(videos),
    if mod(i,100)==0, fprintf('Reading results %d/%d\n', i, numel(videos)); end
    lbl_path = [labels_dir videos{i} '.txt']; 
    lbl = csvread( lbl_path ); % SIFTflow distance
    for j=1:numel(models),
        pred_path = sprintf('%s/%s.txt',pred_dir{j},videos{i});
        pred = csvread( pred_path ); % Prediction result
        % MATLAB sometimes ignores the last frame of video; we cap the max frame index
        pred = max(1,min(pred,numel(lbl))); 
        for k=1:numel(theta),
            gt_lbl = lbl<=theta(k);
            match = gt_lbl(pred);
            for l=1:topK,
                precision_k(i,j,k,l) = sum(match(1:min(l,numel(pred))))>0;
            end
        end
    end
end

% Perform significance test
significance = zeros(numel(models), numel(models), topK);
for i=1:numel(models),
    for j=i+1:numel(models),
        for k=1:topK,
            ri = precision_k(:,i,find(theta==0.005),k);
            rj = precision_k(:,j,find(theta==0.005),k);
            [~,significance(i,j,k)] = ttest2( ri, rj );
            significance(j,i,k) = significance(i,j,k);
        end
    end
end


%% Print out results
for i=1:numel(models),
    fprintf('%30s:\t',models{i});
    for j=1:topK,
        fprintf('%.4f\t',mean(precision_k(:,i,end,j)));
    end
    fprintf('\n');
end

print_significance_test_results = false;
if print_significance_test_results,
    significance_test_bar = 0.05;
    for reference_model_index=[8],
        fprintf('\nStatistical significance (reference: %s)\n', ....
            models{reference_model_index});
        for i=1:numel(models),
            if i==reference_model_index, continue; end
            fprintf('%30s:\t', models{i});
            for j=[1 3 5],
                pvalue = significance(reference_model_index,i,j);
                if pvalue<=significance_test_bar,
                    fprintf('(k=%d) p=%f*\t', j, pvalue);
                else
                    fprintf('(k=%d) p=%f\t', j, pvalue);
                end
                if j==5, fprintf('\n'); end
            end
        end
    end
end


%% Detailed results for paper
generate_paper_results = true;
save_figure = false;
if generate_paper_results,
    
    % 
    % Table 1: Aggregated performance score. Set P@k=1,3,5,10, theta=0.005,
    %
    latex_model_name={'Random','K-means Centroid','K-means Stillness',...
        'G.-LASSO~\cite{cong2012towards}','Beauty Rank',...
        'CNN~\cite{yang2015blog}','Ours Supervised','Ours Unsupervised'};
    PatK = zeros(numel(models),topK);
    for i=1:numel(models),
        for j=1:topK,
            PatK(i,j) = mean(precision_k(:,i,find(theta==0.005),j));
        end
    end

    reference_model_index = 8; % significance wrt 'Ours Unsupervised'
    [~,sort_idx] = sort(PatK,'descend');
    for i=1:numel(latex_model_name)
        fprintf('\t\t%s & ',latex_model_name{i});
        for j=[1 3 5],
            pvalue = significance(reference_model_index, i, j);
            mark='';
            if pvalue < 0.001,
                mark='$^{\ddagger}$';
            elseif pvalue < 0.05,
                mark='$^{\dagger}$';
            end
            if i==sort_idx(1,j)
                fprintf('\\textbf{%.4f}% ',PatK(i,j));
            elseif i==sort_idx(2,j),
                fprintf('\\underline{%.4f%s} ',PatK(i,j), mark);
            else
                fprintf('%.4f%s ',PatK(i,j), mark);
            end
            if j==5,
                if i==6,
                    fprintf('\\\\ \\hline\n');
                else
                    fprintf('\\\\ \n');
                end
            else
                fprintf('& ');
            end
        end
    end

    %
    % Figure. Full spectrum of P@k
    %
    figure
    r = reshape(mean(precision_k(:,:,:,topK),1), [numel(models) numel(theta)])';
    h = plot(r);
    legend( model_names, 'Location','NorthWest');
    title( sprintf('Precision@K (K=5)') );
    ylabel('Mean Precision@k');
    xlabel('SIFTflow dist threshold (\theta)');
    xlim([1 numel(theta)]);
    set(gca,'XTick',0:10:50);
    set(gca,'XTickLabel',{'0.000','0.001','0.002','0.003','0.004','0.005'});
    set(gcf,'Position',[1290 609 370 340]);
    
    my_linestyle = {'-.','--','--','-.',':','-.','-','-'};
    my_linecolor = [4 15 141; 10 52 245; 34 175 247; 78 254 192; ...
                    0 0 0; 253 158 40; 252 23 26; 126 3 8] / 255;
    my_linewidth = ones(1,numel(models))*2;
    for i=1:numel(h),
        h(i).Color = my_linecolor(i,:);
        h(i).LineStyle = my_linestyle{i};
        h(i).LineWidth = my_linewidth(i);
    end
    
    if save_figure ,
        addpath(genpath(export_fig_path));
        export_fig(sprintf('./prec_k_cikm2016.eps'),'-eps','-transparent');
    end

    
    %
    % Figure. Bar plot per channel. Set P@5, theta=0.005
    %
    figure;
    xtick_name = {'Auto','Celeb','Comedy','Cute','Fashion','Finance','Food',...
        'Game','Health','Intl.','Makers','Movie','News','Parent','Sports',...
        'TV','Tech.','Travel','','All'};

    categories_unique = unique(categories);
    p_k_per_category = zeros(numel(categories_unique),numel(models));
    for i=1:numel(categories_unique),
        vids = strcmp(categories,categories_unique{i});
        p_k_per_category(i,:) = mean(precision_k(vids,:,find(theta==0.005),topK),1);
    end
    p_k_per_category = [p_k_per_category; zeros(1,numel(models)); ...
                        mean(precision_k(:,:,find(theta==0.005),topK))];
    
    colormap(jet);
    hbar = bar( p_k_per_category );
    legend(model_names,'orientation','vertical');
    set(gcf,'Position',[1 549 1680 303]);
    set(gca,'YGrid','on');
    set(gca,'XTick',1:numel(xtick_name),'XTickLabel',xtick_name);
    xlim([0 numel(xtick_name)+1]);
    ylim([0 0.6]);
    title('Precision@K per channel (K=5, \theta=0.005)');

    if save_figure,
        addpath(genpath(export_fig_path));
        export_fig(sprintf('./prec_k_channel_cikm2016.eps'),'-eps','-transparent');
    end

end
