% DISEM – Intelligent Diagnosis of Electromechanical Systems
% Authors: Youness LAAZIZ, Mamadou Bakary KEBE
% Supervisor: Prof. Mohamed RAFIK
% ENSET – 2025/2026
%
% Academic project – proper citation required.
% GitHub: https://github.com/laazizyounessgesii-ops/DISEM-End-to-End-Embedded-AI



function generate_labels_csv_3500(folder)
% Génère labels.csv: mapping fichier -> class_id, sample_id
%
% Usage: generate_labels_csv_3500("C:\...\dataset_csv_3500")

    if nargin < 1 || strlength(folder)==0
        error('Donne le dossier: generate_labels_csv_3500("C:\\...\\dataset_csv_3500")');
    end

    files = dir(fullfile(folder, 'C*_*.csv'));
    if isempty(files)
        error('Aucun CSV trouvé dans: %s', folder);
    end

    filenames = strings(numel(files),1);
    class_id  = zeros(numel(files),1);
    sample_id = zeros(numel(files),1);

    for k = 1:numel(files)
        filenames(k) = string(files(k).name);

        tok = regexp(files(k).name, '^C(\d+)_(\d+)\.csv$', 'tokens', 'once');
        if isempty(tok)
            error('Nom de fichier inattendu: %s (attendu: C<id>_<sid>.csv)', files(k).name);
        end

        class_id(k)  = str2double(tok{1});
        sample_id(k) = str2double(tok{2});
    end

    % Tri propre
    TT = table(filenames, class_id, sample_id);
    TT = sortrows(TT, {'class_id','sample_id'});

    outpath = fullfile(folder, 'labels.csv');
    writetable(TT, outpath);

    fprintf('OK: labels.csv généré (%d lignes) -> %s\n', height(TT), outpath);
end
