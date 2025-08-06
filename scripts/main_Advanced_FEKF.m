%% ----------------------------
% Advanced FEKF Implementation
% Input: Work_mode: Mode of working condition 1 --> BBDST, 2 --> constant current
%                   SOC_est_init: The initial value of estimated SOC      
%% ----------------------------
function main_Advanced_FEKF(Work_mode, SoC_est_init)
    if nargin == 0  % Set parameter by default
        Work_mode = 1;
        SoC_est_init = 1;
    elseif nargin == 1
        SoC_est_init = 1;
    end
    if Work_mode == 1
        sim BBDST_workingcondition;
        I = -(current.data)' * 1.5 / 50;
    elseif Work_mode == 2
        N = 60001;
        I = 1.5 * ones(1, N);
        I(ceil(N / 5) : ceil(N * 3 / 9)) = 0;
        I(ceil(N * 5 / 9) : ceil(N * 4 / 5)) = 0;
    else
        disp("Input error!");
        disp("Work_mode: Mode of working condition");
        disp("           1 --> BBDST, 2 --> constant current ");
        disp("SOC_est_init : The initial value of estimated SOC");
        return;
    end
    tic;  % start time
    [avr_err_FEKF, std_err_FEKF, SoC_FEKF, Err_FEKF] = FEKF_Advanced(SoC_est_init, I);
    toc;  % end time
    fprintf('Initial SOC : %f\nWorking Mode: %d\n', SoC_est_init, Work_mode);
    fprintf("Advanced FEKF Results:\n");
    fprintf("avr_err_FEKF --> %f\n", avr_err_FEKF);
    fprintf("standard_err_FEKF --> %f\n", std_err_FEKF);
end