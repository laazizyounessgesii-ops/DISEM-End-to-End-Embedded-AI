% DISEM – Intelligent Diagnosis of Electromechanical Systems
% Authors: Youness LAAZIZ, Mamadou Bakary KEBE
% Supervisor: Prof. Mohamed RAFIK
% ENSET – 2025/2026
%
% Academic project – proper citation required.
% GitHub: https://github.com/laazizyounessgesii-ops/DISEM-End-to-End-Embedded-AI



function meta = configurer_defaut_PSC_matlab(classe_id, echantillon_id)
%CONFIGURER_DEFAUT_PSC_MATLAB
% Retourne une struct meta contenant paramètres moteur, alim et vibration.

    if nargin < 2
        error('Usage: meta=configurer_defaut_PSC_matlab(classe_id, echantillon_id)');
    end
    if classe_id < 1 || classe_id > 7
        error('classe_id doit être 1..7');
    end

    seed = echantillon_id*100 + classe_id;
    rng(seed);

    % ===== Paramètres moteur (sain) =====
    p0.Rm = 10;    p0.Lm = 0.1;
    p0.Ra = 12;    p0.La = 0.08;
    p0.Ke = 0.15;  p0.Kt = 0.035;
    p0.J  = 0.002; p0.F  = 0.001;
    p0.ws = 314.16;
    p0.Tl = 2.0;

    % ===== Alimentation PSC =====
    Vm0 = 311;
    Va_ratio = 0.8;
    Vm_phase = 0;
    Va_phase = 80*pi/180;

    % ===== Base vibration =====
    vb0.gain_bruit = 0.05;
    vb0.gain_HP_Te = 0.30;
    vb0.gain_wr    = 0.50;

    vb0.gain_mis_1x = 0;
    vb0.gain_mis_2x = 0;

    vb0.gain_impact = 0;
    vb0.f_imp = 0;
    vb0.f0  = 900;
    vb0.tau = 0.002;

    % ===== Init =====
    p = p0;

    k_bruit = 1.0; k_HP = 1.0; k_wr = 1.0;
    mis1 = 0; mis2 = 0;
    imp  = 0; f_imp = 0; f0 = vb0.f0; tau = vb0.tau;

    switch classe_id
        case 1 % C1 sain
            % rien

        case 2 % C2 court-circuit partiel stator (proxy)
            sev = 0.15 + 0.35*rand();
            p.Rm = p0.Rm*(1 + 0.8*sev);
            p.Lm = p0.Lm*(1 - 0.4*sev);
            k_HP = 1.5;

        case 3 % C3 déséquilibre statorique
            sev = 0.15 + 0.35*rand();
            p.Ra = p0.Ra*(1 + 1.0*sev);
            p.La = p0.La*(1 - 0.3*sev);
            k_HP = 1.3;
            k_wr = 1.2;

        case 4 % C4 rotor (proxy ripple couple)
            k_HP = 1.8;
            k_wr = 1.2;
            p.Kt = p0.Kt*(1 + 0.10*rand());

        case 5 % C5 désalignement
            k_wr = 1.2;
            mis1 = 40*(0.8 + 0.4*rand());
            mis2 = 15*(0.8 + 0.4*rand());

        case 6 % C6 roulement (dominant)
            k_bruit = 1.1;

            % impacts nets et détectables via énergie HF
            imp  = 420*(0.8 + 0.4*rand());       % ~336..588
            f_imp = 90 + 50*rand();               % 90..140 Hz
            f0  = 950 + 250*rand();               % 950..1200 Hz
            tau = 0.0016 + 0.0008*rand();         % 1.6..2.4 ms

        case 7 % C7 combiné raisonnable
            sev = 0.15 + 0.35*rand();
            p.Rm = p0.Rm*(1 + 0.8*sev);
            p.Lm = p0.Lm*(1 - 0.4*sev);

            k_bruit = 1.25; k_HP = 1.7; k_wr = 1.25;

            mis1 = 22*(0.8 + 0.4*rand());
            mis2 = 0;

            imp  = 110*(0.8 + 0.4*rand());       % modéré
            f_imp = 18 + 18*rand();               % 18..36 Hz
            f0  = 850 + 180*rand();               % 850..1030 Hz
            tau = 0.0010 + 0.0007*rand();         % 1.0..1.7 ms
    end

    % ===== Nuisances communes =====
    Vm_ampl = Vm0*(0.95 + 0.10*rand());
    Va_ampl = Va_ratio*Vm_ampl;

    p.Tl = p.Tl*(0.85 + 0.30*rand());

    cap_bruit = (0.85 + 0.30*rand());
    cap_HP    = (0.90 + 0.20*rand());
    cap_wr    = (0.90 + 0.20*rand());

    vb.gain_bruit = vb0.gain_bruit * k_bruit * cap_bruit;
    vb.gain_HP_Te = vb0.gain_HP_Te * k_HP    * cap_HP;
    vb.gain_wr    = vb0.gain_wr    * k_wr    * cap_wr;

    vb.gain_mis_1x = mis1;
    vb.gain_mis_2x = mis2;

    vb.gain_impact = imp;
    vb.f_imp = f_imp;
    vb.f0  = f0;
    vb.tau = tau;

    meta = struct();
    meta.classe_id = classe_id;
    meta.echantillon_id = echantillon_id;
    meta.seed = seed;

    meta.params_moteur = p;

    meta.Vm_ampl = Vm_ampl;
    meta.Va_ampl = Va_ampl;
    meta.Vm_phase = Vm_phase;
    meta.Va_phase = Va_phase;

    meta.vibration = vb;

    fprintf('C%d | gb=%.3f gHP=%.3f gwr=%.3f | mis1=%.1f mis2=%.1f | imp=%.1f fimp=%.1fHz f0=%.0fHz tau=%.1fms | Vm=%.1f Va=%.1f\n',...
        classe_id, vb.gain_bruit, vb.gain_HP_Te, vb.gain_wr, vb.gain_mis_1x, vb.gain_mis_2x, ...
        vb.gain_impact, vb.f_imp, vb.f0, 1e3*vb.tau, Vm_ampl, Va_ampl);
end
