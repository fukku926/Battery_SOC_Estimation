%% ==================================================================
%  Description: SoC estimation using EKF/UKF/FEKF based on the Thevenin model
%  Input: SoC_upd_init, the initial updated value of SoC
%         Current, the working condition current generated from BBDST工况.slx
%% ==================================================================

function [avr_err_EKF, std_err_EKF, avr_err_UKF, std_err_UKF, avr_err_FEKF, std_err_FEKF] = EKF_UKF_FEKF_Thev(SoC_upd_init, current)
    %% Initialization -----------------------------------------------
    ts  = 1;  % sample interval
    tr = 0.1;  % smallest time interval used to simulate the real SOC
    N = 5000;
    Capacity = 1.5;
    SoC_real(1, 1) = 1;  % Initial real SoC value
    States_real = [SoC_real(1, 1); 0];  % (SoC_real, Up_real)
    States_upd = [SoC_upd_init; 0];  % (SOC_upd, Up_upd)
    SoC_AH(1,1) = SoC_upd_init;  % Initial value of AH
    SoC_EKF(1,1) = SoC_upd_init;
    SoC_UKF(:,1) = SoC_upd_init;
    SoC_FEKF(1,1) = SoC_upd_init;  % FEKF SoC estimation
    
    % Errors
    Err_EKF = zeros(1, N);
    Err_UKF = zeros(1, N);
    Err_AH = zeros(1, N);
    Err_FEKF = zeros(1, N);  % FEKF error
    Err_EKF(1,1) = SoC_real(1,1) - States_upd(1,1);  % Error of EKF
    Err_UKF(1,1) = SoC_real(1,1) - SoC_UKF(1,1);  % Error of UKF
    Err_AH(1,1) = SoC_real(1,1) - SoC_AH(1,1);  % Error of AH
    Err_FEKF(1,1) = SoC_real(1,1) - SoC_FEKF(1,1);  % Error of FEKF

    % EKF parameters
    P_Cov = [1e-8 0; 0 1e-6];  % covariance matrix
    Qs = 4e-9;  % variance of the SoC process noise, also for UKF
    Qu = 1e-8;  % variance of the Up process noise
    R = 1e-6;  % variance of observation noise, also for UKF
    I_real = current;
    
    % FEKF parameters
    P_Cov_FEKF = [1e-8 0; 0 1e-6];  % FEKF covariance matrix
    Qs_FEKF = 4e-9;  % FEKF SoC process noise
    Qu_FEKF = 1e-8;  % FEKF Up process noise
    R_FEKF = 1e-6;  % FEKF observation noise
    feedback_gain = 0.1;  % Feedback gain parameter for FEKF
    
    % UKF parameters
    n = 1;  % dimension
    alpha = 0.04;
    beta = 2;
    lambda = 1.5;    
    % weight
    Wm = [lambda / (n + lambda), 0.5 / (n + lambda) + zeros(1, 2 * n)];
    Wc = Wm;
    Wc(1) = Wc(1) + (1 - alpha^2 + beta);
    P = 1e-6; % initial variance value of state error

    % SoC estimation process  ---------------------------------------
    for T = 2 : N
        %% Simulating the real states -------------------------------
        for t = (T-1) * ts/tr - (ts/tr - 2) : (T-1) * ts/tr + 1
            Rp = 0.02346-0.10537 * SoC_real(1, t-1)^1 + 1.1371 * SoC_real(1, t-1)^2 - 4.55188 * SoC_real(1, t-1)^3 + 8.26827 * SoC_real(1, t-1)^4 - 6.93032 * SoC_real(1,t-1)^5 + 2.1787 * SoC_real(1, t-1)^6;
            Cp = 203.1404 + 3522.78847 * SoC_real(1, t-1) - 31392.66753 * SoC_real(1, t-1)^2 + 122406.91269 * SoC_real(1, t-1)^3 - 227590.94382 * SoC_real(1, t-1)^4 + 198281.56406 * SoC_real(1, t-1)^5 - 65171.90395 * SoC_real(1, t-1)^6;
            tao = Rp * Cp;
            
            A2 = exp(-tr / tao);
            A =[1 0; 0 A2];  % State transformation matrix
            B1 = - tr / (Capacity * 3600);
            B2 = Rp * (1 - exp(-tr / tao));
            B = [B1; B2];  % Input control matrix
            
            States_real(:, t) = A * States_real(:, t-1) + B * I_real(1, t) + [sqrt(Qs) * randn; sqrt(Qu) * randn];
            SoC_real(1, t) = States_real(1, t);
        end
        UOC_real = 3.44003 + 1.71448 * States_real(1, t) - 3.51247 * States_real(1, t)^2  + 5.70868 * States_real(1, t)^3 - 5.06869 * States_real(1, t)^4 + 1.86699 * States_real(1, t)^5;
        Rint_real = 0.04916 + 1.19552 * States_real(1, t) - 6.25333 * States_real(1, t)^2 + 14.24181 * States_real(1, t)^3 - 13.93388 * States_real(1, t)^4 + 2.553 * States_real(1, t)^5 + 4.16285 * States_real(1, t)^6 - 1.8713 * States_real(1, t)^7;
        % Observed voltage/current with observation error
        UL_ob_EKF = UOC_real - States_real(2, t) - I_real(1, t) * Rint_real + sqrt(R) * randn;
        UL_ob_UKF = UOC_real - B2 * I_real(1,t) - I_real(1,t) * Rint_real + sqrt(R)*randn;
        UL_ob_FEKF = UOC_real - States_real(2, t) - I_real(1, t) * Rint_real + sqrt(R_FEKF) * randn;  % FEKF observation
        I_ob = I_real(t) + (0.01 * Capacity) * randn;  % observation error

        %% AH process -----------------------------------------------
        SoC_AH(1, T) = SoC_AH(1, T-1) - ts / (Capacity * 3600) * I_ob;

        %% EKF process ----------------------------------------------
        % predict
        Rp = 0.02346 - 0.10537 * SoC_EKF(1, T-1)^1 + 1.1371 * SoC_EKF(1,T-1)^2 - 4.55188 * SoC_EKF(1,T-1)^3 + 8.26827 * SoC_EKF(1, T-1)^4 - 6.93032 * SoC_EKF(1,T-1)^5 + 2.1787 * SoC_EKF(1, T-1)^6;
        Cp = 203.1404 + 3522.78847 * SoC_EKF(1, T-1) - 31392.66753 * SoC_EKF(1, T-1)^2 + 122406.91269 * SoC_EKF(1, T-1)^3 - 227590.94382 * SoC_EKF(1, T-1)^4 + 198281.56406 * SoC_EKF(1, T-1)^5 - 65171.90395 * SoC_EKF(1, T-1)^6;
        tao = Rp * Cp;
        A = [1 0; 0 exp(-ts / tao)];  % State transformation matrix
        B = [-ts / (Capacity * 3600); Rp * (1 - exp(-ts / tao))];  % Input control matrix
        States_pre = A * States_upd(:, T - 1) + B * I_ob;  % states prediction
        SoC_pre = States_pre(1, 1);  % predicted value of SoC
        Up_pre = States_pre(2, 1);  % predicted value of the polarization voltage
        P_Cov = A * P_Cov * A' + [Qs 0; 0 Qu];
        UOC_pre = 3.44003 + 1.71448 * SoC_pre - 3.51247 * SoC_pre^2 + 5.70868 * SoC_pre^3 - 5.06869 * SoC_pre^4 + 1.86699 * SoC_pre^5;
        Ro_pre = 0.04916 + 1.19552 * SoC_pre - 6.25333 * SoC_pre^2 + 14.24181* SoC_pre^3 - 13.93388 * SoC_pre^4 + 2.553 * SoC_pre^5 + 4.16285 * SoC_pre^6 - 1.8713 * SoC_pre^7;
        UL_pre = UOC_pre - Up_pre - I_ob * Ro_pre;
        % linearization
        C1 = 1.71448 - 2 * 3.51247 * SoC_EKF(1,T-1) + 3 * 5.70868 * SoC_EKF(1, T-1)^2 - 4 * 5.06869 * SoC_EKF(1, T-1)^3 + 5 * 1.86699 * SoC_EKF(1, T-1)^4;
        C = [C1 -1];
        % update
        K = P_Cov * C' * (C * P_Cov * C' + R)^(-1);  % gain
        States_upd(:, T) = States_pre + K * (UL_ob_EKF - UL_pre);
        P_Cov = P_Cov - K * C * P_Cov;
        SoC_EKF(1, T) = States_upd(1, T);
        
        %% FEKF process ----------------------------------------------
        % FEKF prediction (similar to EKF but with feedback mechanism)
        Rp_FEKF = 0.02346 - 0.10537 * SoC_FEKF(1, T-1)^1 + 1.1371 * SoC_FEKF(1,T-1)^2 - 4.55188 * SoC_FEKF(1,T-1)^3 + 8.26827 * SoC_FEKF(1, T-1)^4 - 6.93032 * SoC_FEKF(1,T-1)^5 + 2.1787 * SoC_FEKF(1, T-1)^6;
        Cp_FEKF = 203.1404 + 3522.78847 * SoC_FEKF(1, T-1) - 31392.66753 * SoC_FEKF(1, T-1)^2 + 122406.91269 * SoC_FEKF(1, T-1)^3 - 227590.94382 * SoC_FEKF(1, T-1)^4 + 198281.56406 * SoC_FEKF(1, T-1)^5 - 65171.90395 * SoC_FEKF(1, T-1)^6;
        tao_FEKF = Rp_FEKF * Cp_FEKF;
        A_FEKF = [1 0; 0 exp(-ts / tao_FEKF)];  % State transformation matrix
        B_FEKF = [-ts / (Capacity * 3600); Rp_FEKF * (1 - exp(-ts / tao_FEKF))];  % Input control matrix
        
        % FEKF state prediction with feedback
        States_pre_FEKF = A_FEKF * [SoC_FEKF(1, T-1); 0] + B_FEKF * I_ob;
        SoC_pre_FEKF = States_pre_FEKF(1, 1);
        Up_pre_FEKF = States_pre_FEKF(2, 1);
        
        % FEKF covariance prediction
        P_Cov_FEKF = A_FEKF * P_Cov_FEKF * A_FEKF' + [Qs_FEKF 0; 0 Qu_FEKF];
        
        % FEKF measurement prediction
        UOC_pre_FEKF = 3.44003 + 1.71448 * SoC_pre_FEKF - 3.51247 * SoC_pre_FEKF^2 + 5.70868 * SoC_pre_FEKF^3 - 5.06869 * SoC_pre_FEKF^4 + 1.86699 * SoC_pre_FEKF^5;
        Ro_pre_FEKF = 0.04916 + 1.19552 * SoC_pre_FEKF - 6.25333 * SoC_pre_FEKF^2 + 14.24181* SoC_pre_FEKF^3 - 13.93388 * SoC_pre_FEKF^4 + 2.553 * SoC_pre_FEKF^5 + 4.16285 * SoC_pre_FEKF^6 - 1.8713 * SoC_pre_FEKF^7;
        UL_pre_FEKF = UOC_pre_FEKF - Up_pre_FEKF - I_ob * Ro_pre_FEKF;
        
        % FEKF linearization
        C1_FEKF = 1.71448 - 2 * 3.51247 * SoC_FEKF(1,T-1) + 3 * 5.70868 * SoC_FEKF(1, T-1)^2 - 4 * 5.06869 * SoC_FEKF(1, T-1)^3 + 5 * 1.86699 * SoC_FEKF(1, T-1)^4;
        C_FEKF = [C1_FEKF -1];
        
        % FEKF update with feedback mechanism
        K_FEKF = P_Cov_FEKF * C_FEKF' * (C_FEKF * P_Cov_FEKF * C_FEKF' + R_FEKF)^(-1);  % FEKF gain
        
        % Feedback mechanism: adjust gain based on innovation
        innovation_FEKF = UL_ob_FEKF - UL_pre_FEKF;
        feedback_factor = 1 + feedback_gain * abs(innovation_FEKF);
        K_FEKF = K_FEKF * feedback_factor;  % Apply feedback to gain
        
        % FEKF state update
        States_upd_FEKF = States_pre_FEKF + K_FEKF * innovation_FEKF;
        P_Cov_FEKF = P_Cov_FEKF - K_FEKF * C_FEKF * P_Cov_FEKF;
        SoC_FEKF(1, T) = States_upd_FEKF(1, 1);
        
        %% UKF process ----------------------------------------------
        Xsigma = SoC_UKF(T - 1);
        pk = sqrt((n + lambda) * P);
        % sigma sampling points determination
        sigma1 = zeros(1, n);
        sigma2 = zeros(1, n);
        for i = 1 : n
           sigma1(i) = Xsigma + pk;
           sigma2(i) = Xsigma - pk;
        end
        sigma = [Xsigma sigma1 sigma2];
        % predict State (SoC)
        sxk = 0;  % Mean value of state
        for ks = 1 : 2*n+1
            sigma(ks) = sigma(ks) - I_ob * ts / (Capacity * 3600);
            sxk = Wm(ks) * sigma(ks) + sxk;         
        end
        % predict the variance
        spk = 0;
        for kp = 1 : 2*n+1
            spk = Wc(kp) * (sigma(kp) - sxk) * (sigma(kp) - sxk)' + spk;
        end
        spk = spk + Qs;
        % update sigma sampling points
        pkr = sqrt((n + lambda) * spk);
        for k = 1 : n
           sigma1(k) = sxk + pkr;
           sigma2(k) = sxk - pkr;
        end
        Resigma = [sxk sigma1 sigma2];
        % predict UL
        gamma = zeros(1, 2 * n + 1);
        for i = 1 : 2*n+1
            UOC_pre = 3.44003 + 1.71448 * Resigma(i) - 3.51247 * Resigma(i)^2 + 5.70868 * Resigma(i)^3 - 5.06869 * Resigma(i)^4 + 1.86699 * Resigma(i)^5;
            Ro_pre = 0.04916 + 1.19552 * Resigma(i) - 6.25333 * Resigma(i)^2 + 14.24181 * Resigma(i)^3 - 13.93388 * Resigma(i)^4 + 2.553 * Resigma(i)^5 + 4.16285 * Resigma(i)^6 - 1.8713 * Resigma(i)^7;
            Rp = 0.02346 - 0.10537 * Resigma(i)^1 + 1.1371 * Resigma(i)^2 - 4.55188 * Resigma(i)^3 + 8.26827 * Resigma(i)^4 - 6.93032*Resigma(i)^5 + 2.1787 * Resigma(i)^6;
            Cp = 203.1404 + 3522.78847 * Resigma(i) - 31392.66753 * Resigma(i)^2 + 122406.91269 * Resigma(i)^3 - 227590.94382 * Resigma(i)^4 + 198281.56406 * Resigma(i)^5 - 65171.90395 * Resigma(i)^6;
            tao = Rp * Cp;
            gamma(i) = UOC_pre - I_ob * Ro_pre - I_ob * Rp * (1 - exp(-ts/tao));
        end
        syk = 0;  % Mean value of observed UL
        for i = 1 : 2*n+1
            syk = syk + Wm(i) * gamma(i);
        end
        pyy = 0;  % Calculate the variance of the observation error
        for i = 1 : 2*n+1
            pyy = Wc(i) * (gamma(i) - syk) * (gamma(i) - syk)' + pyy;
        end
        pyy = pyy + R;
        % covariance matrix of error
        pxy = 0;  % describe the relationship between the state and the observation
        for i = 1 : 2*n+1
            pxy = Wc(i) * (Resigma(i) - sxk) * (gamma(i) - syk)' + pxy;
        end
        % update
        kgs = pxy / pyy;  % correction factor, namely kalman gain
        SoC_UKF(T) = sxk + kgs * (UL_ob_UKF - syk);  % update the state SoC
        P = spk - kgs * pyy * kgs';  % update the variance P
        
        %% Error ----------------------------------------------------
        Err_AH(1, T) = SoC_real(1, t) - SoC_AH(1, T);
        Err_EKF(1, T) = SoC_real(1, t) - SoC_EKF(1, T);
        Err_UKF(1, T) = SoC_real(1, t) - SoC_UKF(T);
        Err_FEKF(1, T) = SoC_real(1, t) - SoC_FEKF(1, T);  % FEKF error
    end 
    
    avr_err_EKF = mean(Err_EKF);
    std_err_EKF = std(Err_EKF,0);
    avr_err_UKF = mean(Err_UKF);
    std_err_UKF = std(Err_UKF,0);
    avr_err_FEKF = mean(Err_FEKF);  % FEKF average error
    std_err_FEKF = std(Err_FEKF,0);  % FEKF standard error
    
    %% Display ------------------------------------------------------
    T = 1 : N;
    figure;
    subplot(2,1,1);
    plot(T, SoC_real(1, 1:(ts/tr):(N*ts/tr-1)), 'LineWidth',2);
    hold on;
    plot(T, SoC_AH(1, 1:N), '-.m', T, SoC_EKF(1,1:N), '-.g');
    plot(T, SoC_UKF(1:N), '-.', 'Color', [1 0.5 0]);
    plot(T, SoC_FEKF(1,1:N), '-.b', 'LineWidth', 1.5);  % FEKF plot
    grid on;
    xlabel('t(s)');
    ylabel('SOC');
    legend('SoC_{Real}', 'SoC_{AH}', 'SoC_{EKF}', 'SoC_{UKF}', 'SoC_{FEKF}');
    subplot(2,1,2);
    plot(T, Err_AH(1, 1:N), '-r', T, Err_EKF(1,1:N), '-.g');
    hold on;
    plot(T, Err_UKF(1,1:N), '-.', 'Color', [1 0.5 0]);
    plot(T, Err_FEKF(1,1:N), '-.b', 'LineWidth', 1.5);  % FEKF error plot
    grid on;
    xlabel('t(s)');
    ylabel('error');
    legend('Err_{AH}', 'Err_{EKF}', 'Err_{UKF}', 'Err_{FEKF}', 'Location', 'Best');
end