% DISEM – Intelligent Diagnosis of Electromechanical Systems
% Authors: Youness LAAZIZ, Mamadou Bakary KEBE
% Supervisor: Prof. Mohamed RAFIK
% ENSET – 2025/2026
%
% Academic project – proper citation required.
% GitHub: https://github.com/laazizyounessgesii-ops/DISEM-End-to-End-Embedded-AI


function vib = vibration_model_matlab(Te, wr, vibp, Fs, seed)
%VIBRATION_MODEL_MATLAB
% - Base: bruit + HP(Te) + dwr + misalignment
% - Bearing: impacts "quasi-périodiques" + jitter + résonateur amorti (f0,tau)
% Objectif:
%   C6 -> nb_imp élevé et réparti, kurtosis global élevée MAIS pas "explosive"
%   C7 -> impacts modérés, raisonnable
%
% Entrées:
%   Te   Nx1
%   wr   Nx1
%   vibp struct : gain_bruit, gain_HP_Te, gain_wr, gain_mis_1x, gain_mis_2x,
%                 gain_impact, f_imp, f0, tau
%   Fs   Hz
%   seed int

    if nargin < 5 || isempty(seed), seed = 1; end
    if nargin < 4 || isempty(Fs), error('Fs requis'); end
    if nargin < 3 || isempty(vibp), vibp = struct(); end

    Te = Te(:); wr = wr(:);
    N  = numel(Te);
    if numel(wr) ~= N
        error('Te et wr doivent avoir la même taille.');
    end

    Ts = 1/Fs;
    t  = (0:N-1)'*Ts;

    rng(seed,'twister');

    % -------- helper field safe --------
    getf = @(s,name,def) local_getfielddef(s,name,def);

    % -------- base params --------
    gB  = getf(vibp,'gain_bruit', 0);
    gHP = getf(vibp,'gain_HP_Te',0);
    gWR = getf(vibp,'gain_wr',   0);
    gM1 = getf(vibp,'gain_mis_1x',0);
    gM2 = getf(vibp,'gain_mis_2x',0);

    % bearing params
    gImp = getf(vibp,'gain_impact',0);
    fImp = getf(vibp,'f_imp',0);
    f0   = getf(vibp,'f0', 1000);
    tau  = getf(vibp,'tau',0.002);

    % -------- BASE vibration --------
    n = randn(N,1);

    % HP(Te) stable
    fc = 20;
    a  = exp(-2*pi*fc*Ts);
    hp = zeros(N,1);
    for k=2:N
        hp(k) = a*(hp(k-1) + Te(k) - Te(k-1));
    end

    dwr   = [0; diff(wr)]/Ts;
    theta = cumsum(wr)*Ts;

    vib_base = ...
        gB  * n + ...
        gHP * hp + ...
        gWR * dwr + ...
        gM1 * sin(theta) + ...
        gM2 * sin(2*theta);

    % -------- BEARING (impacts distribués) --------
    vib_bearing = zeros(N,1);

    if gImp > 0 && fImp > 0 && tau > 0 && f0 > 0

        % période impacts
        Timp = max(round(Fs / fImp), 8);

        % forcer une distribution sur tout le signal (après 0.2s)
        startK = 1 + round(0.2*Fs);
        idx_events = [];

        k = startK;
        while k <= N
            jitter = round(0.20*Timp*(2*rand-1));     % jitter +-20%
            k2 = k + max(6, Timp + jitter);
            idx_events(end+1,1) = k; %#ok<AGROW>
            k = k2;
        end

        % amplitudes bornées; évite "un impact géant" qui explose KurtG
        % lognormal léger + clamp
        amp = gImp .* exp(0.10*randn(numel(idx_events),1));
        amp = min(max(amp, 0.60*gImp), 1.40*gImp);

        sgn = sign(randn(numel(idx_events),1));
        u = zeros(N,1);
        u(idx_events) = amp .* sgn;

        % résonateur amorti
        r  = exp(-(1/Fs)/tau);
        w0 = 2*pi*f0/Fs;

        y1 = 0; y2 = 0;
        for k=3:N
            y  = 2*r*cos(w0)*y1 - (r^2)*y2 + u(k);
            y2 = y1; y1 = y;
            vib_bearing(k) = y;
        end
    end

    % -------- MIX (C6 dominant / C7 modéré) --------
    if gImp <= 0
        vib = vib_base;
    else
        if gImp > 150
            % C6 : bearing dominant mais on garde un peu de base
            vib = 0.35*vib_base + 1.00*vib_bearing;
        else
            % C7 : modéré (très important)
            vib = 1.00*vib_base + 0.20*vib_bearing;
        end
    end

    vib = vib - mean(vib);

end

% ===== helper =====
function v = local_getfielddef(s, name, def)
    if isstruct(s) && isfield(s,name)
        tmp = s.(name);
        if ~isempty(tmp) && all(isfinite(tmp))
            v = tmp;
            return;
        end
    end
    v = def;
end
