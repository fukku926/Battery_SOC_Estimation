%% ==================================================================
%  Description: Advanced Feedback Extended Kalman Filter (FEKF) for SoC estimation
%  Based on the new paper reference with enhanced feedback mechanisms
%  Input: SoC_upd_init, the initial updated value of SoC
%         Current, the working condition current
%% ==================================================================

function [avr_err_FEKF, std_err_FEKF, SoC_FEKF, Err_FEKF] = FEKF_Advanced(SoC_upd_init, current)
    %% Initialization -----------------------------------------------
    ts  = 1;  % sample interval
    tr = 0.1;  % smallest time interval used to simulate the real SOC
    N = 5000;
    Capacity = 1.5;
    SoC_real(1, 1) = 1;  % Initial real SoC value
    States_real = [SoC_real(1, 1); 0];  % (SoC_real, Up_real)
    SoC_FEKF(1,1) = SoC_upd_init;  % FEKF SoC estimation
    
    % FEKF state variables
    States_FEKF = [SoC_upd_init; 0];  % (SOC_FEKF, Up_FEKF)
    
    % Errors
    Err_FEKF = zeros(1, N);
    Err_FEKF(1,1) = SoC_real(1,1) - SoC_FEKF(1,1);

    % FEKF parameters with adaptive feedback
    P_Cov_FEKF = [1e-8 0; 0 1e-6];  % FEKF covariance matrix
    Qs_FEKF = 4e-9;  % FEKF SoC process noise
    Qu_FEKF = 1e-8;  % FEKF Up process noise
    R_FEKF = 1e-6;  % FEKF observation noise
    
    % Advanced feedback parameters
    feedback_gain_initial = 0.1;  % Initial feedback gain
    feedback_gain = feedback_gain_initial;
    innovation_window = 10;  % Window for innovation history
    innovation_history = zeros(1, innovation_window);
    innovation_index = 1;
    
    % Adaptive parameters
    adaptive_factor = 1.0;
    min_adaptive_factor = 0.1;
    max_adaptive_factor = 2.0;
    
    I_real = current;

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
            
            States_real(:, t) = A * States_real(:, t-1) + B * I_real(1, t) + [sqrt(Qs_FEKF) * randn; sqrt(Qu_FEKF) * randn];
            SoC_real(1, t) = States_real(1, t);
        end
        
        % Real voltage calculation
        UOC_real = 3.44003 + 1.71448 * States_real(1, t) - 3.51247 * States_real(1, t)^2  + 5.70868 * States_real(1, t)^3 - 5.06869 * States_real(1, t)^4 + 1.86699 * States_real(1, t)^5;
        Rint_real = 0.04916 + 1.19552 * States_real(1, t) - 6.25333 * States_real(1, t)^2 + 14.24181 * States_real(1, t)^3 - 13.93388 * States_real(1, t)^4 + 2.553 * States_real(1, t)^5 + 4.16285 * States_real(1, t)^6 - 1.8713 * States_real(1, t)^7;
        
        % Observed voltage with observation error
        UL_ob_FEKF = UOC_real - States_real(2, t) - I_real(1, t) * Rint_real + sqrt(R_FEKF) * randn;
        I_ob = I_real(t) + (0.01 * Capacity) * randn;

        %% Advanced FEKF process ------------------------------------
        % FEKF prediction
        Rp_FEKF = 0.02346 - 0.10537 * SoC_FEKF(1, T-1)^1 + 1.1371 * SoC_FEKF(1,T-1)^2 - 4.55188 * SoC_FEKF(1,T-1)^3 + 8.26827 * SoC_FEKF(1, T-1)^4 - 6.93032 * SoC_FEKF(1,T-1)^5 + 2.1787 * SoC_FEKF(1, T-1)^6;
        Cp_FEKF = 203.1404 + 3522.78847 * SoC_FEKF(1, T-1) - 31392.66753 * SoC_FEKF(1, T-1)^2 + 122406.91269 * SoC_FEKF(1, T-1)^3 - 227590.94382 * SoC_FEKF(1, T-1)^4 + 198281.56406 * SoC_FEKF(1, T-1)^5 - 65171.90395 * SoC_FEKF(1, T-1)^6;
        tao_FEKF = Rp_FEKF * Cp_FEKF;
        
        A_FEKF = [1 0; 0 exp(-ts / tao_FEKF)];  % State transformation matrix
        B_FEKF = [-ts / (Capacity * 3600); Rp_FEKF * (1 - exp(-ts / tao_FEKF))];  % Input control matrix
        
        % FEKF state prediction
        States_pre_FEKF = A_FEKF * States_FEKF(:, T-1) + B_FEKF * I_ob;
        SoC_pre_FEKF = States_pre_FEKF(1, 1);
        Up_pre_FEKF = States_pre_FEKF(2, 1);
        
        % FEKF covariance prediction with adaptive factor
        P_Cov_FEKF = A_FEKF * P_Cov_FEKF * A_FEKF' + adaptive_factor * [Qs_FEKF 0; 0 Qu_FEKF];
        
        % FEKF measurement prediction
        UOC_pre_FEKF = 3.44003 + 1.71448 * SoC_pre_FEKF - 3.51247 * SoC_pre_FEKF^2 + 5.70868 * SoC_pre_FEKF^3 - 5.06869 * SoC_pre_FEKF^4 + 1.86699 * SoC_pre_FEKF^5;
        Ro_pre_FEKF = 0.04916 + 1.19552 * SoC_pre_FEKF - 6.25333 * SoC_pre_FEKF^2 + 14.24181* SoC_pre_FEKF^3 - 13.93388 * SoC_pre_FEKF^4 + 2.553 * SoC_pre_FEKF^5 + 4.16285 * SoC_pre_FEKF^6 - 1.8713 * SoC_pre_FEKF^7;
        UL_pre_FEKF = UOC_pre_FEKF - Up_pre_FEKF - I_ob * Ro_pre_FEKF;
        
        % FEKF linearization
        C1_FEKF = 1.71448 - 2 * 3.51247 * SoC_FEKF(1,T-1) + 3 * 5.70868 * SoC_FEKF(1, T-1)^2 - 4 * 5.06869 * SoC_FEKF(1, T-1)^3 + 5 * 1.86699 * SoC_FEKF(1, T-1)^4;
        C_FEKF = [C1_FEKF -1];
        
        % Innovation calculation
        innovation_FEKF = UL_ob_FEKF - UL_pre_FEKF;
        
        % Store innovation in history
        innovation_history(innovation_index) = innovation_FEKF;
        innovation_index = mod(innovation_index, innovation_window) + 1;
        
        % Adaptive feedback mechanism
        if T > innovation_window
            innovation_variance = var(innovation_history);
            innovation_mean = mean(innovation_history);
            
            % Adjust feedback gain based on innovation statistics
            if abs(innovation_mean) > 0.01
                feedback_gain = feedback_gain_initial * (1 + abs(innovation_mean));
            else
                feedback_gain = feedback_gain_initial;
            end
            
            % Adjust adaptive factor based on innovation variance
            if innovation_variance > 1e-4
                adaptive_factor = min(max_adaptive_factor, adaptive_factor * 1.1);
            else
                adaptive_factor = max(min_adaptive_factor, adaptive_factor * 0.95);
            end
        end
        
        % FEKF gain calculation with feedback
        K_FEKF = P_Cov_FEKF * C_FEKF' * (C_FEKF * P_Cov_FEKF * C_FEKF' + R_FEKF)^(-1);
        
        % Apply feedback to gain
        feedback_factor = 1 + feedback_gain * abs(innovation_FEKF);
        K_FEKF = K_FEKF * feedback_factor;
        
        % FEKF state update
        States_FEKF(:, T) = States_pre_FEKF + K_FEKF * innovation_FEKF;
        P_Cov_FEKF = P_Cov_FEKF - K_FEKF * C_FEKF * P_Cov_FEKF;
        SoC_FEKF(1, T) = States_FEKF(1, T);
        
        %% Error calculation -----------------------------------------
        Err_FEKF(1, T) = SoC_real(1, t) - SoC_FEKF(1, T);
    end 
    
    avr_err_FEKF = mean(Err_FEKF);
    std_err_FEKF = std(Err_FEKF,0);
    
    %% Display ------------------------------------------------------
    T = 1 : N;
    figure;
    subplot(2,1,1);
    plot(T, SoC_real(1, 1:(ts/tr):(N*ts/tr-1)), 'LineWidth',2);
    hold on;
    plot(T, SoC_FEKF(1,1:N), '-.b', 'LineWidth', 1.5);
    grid on;
    xlabel('t(s)');
    ylabel('SOC');
    legend('SoC_{Real}', 'SoC_{FEKF}');
    title('Advanced FEKF SoC Estimation');
    
    subplot(2,1,2);
    plot(T, Err_FEKF(1,1:N), '-.b', 'LineWidth', 1.5);
    grid on;
    xlabel('t(s)');
    ylabel('error');
    legend('Err_{FEKF}');
    title('Advanced FEKF Estimation Error');
end