% DISEM – Intelligent Diagnosis of Electromechanical Systems
% Authors: Youness LAAZIZ, Mamadou Bakary KEBE
% Supervisor: Prof. Mohamed RAFIK
% ENSET – 2025/2026
%
% Academic project – proper citation required.
% GitHub: https://github.com/laazizyounessgesii-ops/DISEM-End-to-End-Embedded-AI




function generate_dataset_PSC_csv_3500(folder)
% Génère 500 échantillons par classe (7 classes) => 3500 CSV
% Chaque CSV contient: t, im, vib, class_id, sample_id
%
% Usage: generate_dataset_PSC_csv_3500("C:\...\dataset_csv_3500")


    if nargin < 1 || strlength(folder)==0
        error('Donne un dossier de sortie: generate_dataset_PSC_csv_3500("C:\Users\Gros Info\Desktop\Projet_dinnovation_LAAZIZ_KEBE_M-GESII_Documents\Projet_dinnovation_LAAZIZ_KEBE_M-GESII_Documents\dataset_csv_3500")');
    end
    if ~isfolder(folder), mkdir(folder); end

    Fs       = 10000;
    StopTime = 2.5;

    classes = 1:7;
    N_per_class = 500;

    fprintf('=== Génération dataset: %d classes x %d = %d CSV ===\n', ...
        numel(classes), N_per_class, numel(classes)*N_per_class);
    fprintf('Dossier: %s\n', folder);

    for cid = classes
        for sid = 1:N_per_class
            meta = configurer_defaut_PSC_matlab(cid, sid);

            out  = simulate_PSC_one_matlab(meta, Fs, StopTime);

            t  = mustGetFieldAny(out, {'t','time'}); t = t(:);
            im = mustGetFieldAny(out, {'im','i_m','current','courant','i'}); im = im(:);
            vb = mustGetFieldAny(out, {'vibration','vib','v','vibration_final','vibration_signal'}); vb = vb(:);

            % Sécurité longueurs
            n = min([numel(t), numel(im), numel(vb)]);
            t = t(1:n); im = im(1:n); vb = vb(1:n);

            class_id  = repmat(cid, n, 1);
            sample_id = repmat(sid, n, 1);

            T = table(t, im, vb, class_id, sample_id, ...
                'VariableNames', {'t','im','vib','class_id','sample_id'});

            fname = sprintf('C%d_%04d.csv', cid, sid);
            fpath = fullfile(folder, fname);
            writetable(T, fpath);

            if mod(sid,50)==0
                fprintf('C%d: %d/%d OK\n', cid, sid, N_per_class);
            end
        end
    end

    fprintf('=== OK: CSV générés ===\n');
    fprintf('Maintenant lance: generate_labels_csv_3500("%s")\n', folder);
end

% ===== Helpers =====
function x = getFieldAny(s, names)
    x = [];
    for i = 1:numel(names)
        if isstruct(s) && isfield(s, names{i})
            x = s.(names{i});
            return;
        end
    end
end

function x = mustGetFieldAny(s, names)
    x = getFieldAny(s, names);
    if isempty(x)
        error('Champ introuvable. Cherché: %s', strjoin(names, ', '));
    end
end
