% DISEM – Intelligent Diagnosis of Electromechanical Systems
% Authors: Youness LAAZIZ, Mamadou Bakary KEBE
% Supervisor: Prof. Mohamed RAFIK
% ENSET – 2025/2026
%
% Academic project – proper citation required.
% GitHub: https://github.com/laazizyounessgesii-ops/DISEM-End-to-End-Embedded-AI


function out = simulate_PSC_one_matlab(meta, Fs, StopTime)

    p  = meta.params_moteur;
    vb = meta.vibration;

    Ts = 1/Fs;
    t  = (0:Ts:StopTime).';
    N  = numel(t);

    % Etats init
    x = zeros(3,1);   % [im ia wr]
    x(3) = 0;

    im = zeros(N,1);
    ia = zeros(N,1);
    wr = zeros(N,1);
    Te = zeros(N,1);

    % Alim sinusoïdale
    w = 2*pi*50; % 50 Hz
    Vm = meta.Vm_ampl; Va = meta.Va_ampl;
    phm = meta.Vm_phase; pha = meta.Va_phase;

    for k = 1:N
        tk = t(k);

        vm = Vm*sin(w*tk + phm);
        va = Va*sin(w*tk + pha);

        u.vm = vm;
        u.va = va;

        % Calcul couple au temps courant (avec l'état x courant)
        Te(k) = Te_from_state(x, u, p);

        % Intégration RK4 (sauf dernier point)
        if k < N
            x = rk4_step(@(xx) f_psc_dx(xx, u, p), x, Ts);
        end

        im(k) = x(1);
        ia(k) = x(2);
        wr(k) = x(3);
    end

    % Vibration
    vib = vibration_model_matlab(Te, wr, vb, Fs, meta.seed + 999);

    out.t   = t;
    out.im  = im;
    out.ia  = ia;
    out.wr  = wr;
    out.Te  = Te;
    out.vib = vib;
end

% ============================================================
% Dynamiques moteur: retourne seulement dx = [dim; dia; dwr]
% ============================================================
function dx = f_psc_dx(x, u, p)
    im = x(1); ia = x(2); wr = x(3);

    [dim, dia, ~, dwr] = moteur_PSC_param_correct( ...
        u.vm, u.va, im, ia, wr, p.Tl, ...
        p.Rm, p.Lm, p.Ra, p.La, p.Ke, p.Kt, p.J, p.F, p.ws);

    dx = [dim; dia; dwr];
end

% ============================================================
% Couple Te à partir de l'état courant (sans dériver dx)
% ============================================================
function Te = Te_from_state(x, u, p)
    im = x(1); ia = x(2); wr = x(3);

    [~, ~, Te, ~] = moteur_PSC_param_correct( ...
        u.vm, u.va, im, ia, wr, p.Tl, ...
        p.Rm, p.Lm, p.Ra, p.La, p.Ke, p.Kt, p.J, p.F, p.ws);
end

% ============================================================
% RK4
% ============================================================
function xnext = rk4_step(f, x, h)
    k1 = f(x);
    k2 = f(x + 0.5*h*k1);
    k3 = f(x + 0.5*h*k2);
    k4 = f(x + h*k3);
    xnext = x + (h/6)*(k1 + 2*k2 + 2*k3 + k4);
end
