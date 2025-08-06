%% ==================================================================
%  Comprehensive Comparison of EKF, UKF, and FEKF Methods
%  This script compares the performance of different Kalman filter variants
%  for battery SoC estimation
%% ==================================================================

function Compare_All_Methods(Work_mode, SoC_est_init)
    if nargin == 0  % Set parameter by default
        Work_mode = 1;
        SoC_est_init = 1;
    elseif nargin == 1
        SoC_est_init = 1;
    end
    
    fprintf('=== Battery SoC Estimation Method Comparison ===\n');
    fprintf('Initial SOC: %f\n', SoC_est_init);
    fprintf('Working Mode: %d\n', Work_mode);
    fprintf('==============================================\n\n');
    
    % Generate current profile
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
    
    % Run all methods and compare results
    fprintf('Running EKF, UKF, and FEKF methods...\n');
    
    % Method 1: Original EKF and UKF
    tic;
    [avr_err_EKF, std_err_EKF, avr_err_UKF, std_err_UKF] = EKF_UKF_Thev(SoC_est_init, I);
    time_original = toc;
    
    % Method 2: FEKF with basic feedback
    tic;
    [avr_err_EKF2, std_err_EKF2, avr_err_UKF2, std_err_UKF2, avr_err_FEKF, std_err_FEKF] = EKF_UKF_FEKF_Thev(SoC_est_init, I);
    time_fekf = toc;
    
    % Method 3: Advanced FEKF
    tic;
    [avr_err_FEKF_adv, std_err_FEKF_adv, SoC_FEKF_adv, Err_FEKF_adv] = FEKF_Advanced(SoC_est_init, I);
    time_fekf_adv = toc;
    
    % Display comparison results
    fprintf('\n=== Performance Comparison Results ===\n');
    fprintf('Method\t\t\tAverage Error\tStd Error\tExecution Time\n');
    fprintf('--------------------------------------------------------\n');
    fprintf('EKF\t\t\t%.6f\t%.6f\t%.3f s\n', avr_err_EKF, std_err_EKF, time_original);
    fprintf('UKF\t\t\t%.6f\t%.6f\t%.3f s\n', avr_err_UKF, std_err_UKF, time_original);
    fprintf('FEKF (Basic)\t\t%.6f\t%.6f\t%.3f s\n', avr_err_FEKF, std_err_FEKF, time_fekf);
    fprintf('FEKF (Advanced)\t\t%.6f\t%.6f\t%.3f s\n', avr_err_FEKF_adv, std_err_FEKF_adv, time_fekf_adv);
    
    % Find best performing method
    errors = [avr_err_EKF, avr_err_UKF, avr_err_FEKF, avr_err_FEKF_adv];
    methods = {'EKF', 'UKF', 'FEKF (Basic)', 'FEKF (Advanced)'};
    [min_error, best_idx] = min(abs(errors));
    
    fprintf('\n=== Best Performing Method ===\n');
    fprintf('Method: %s\n', methods{best_idx});
    fprintf('Average Error: %.6f\n', errors(best_idx));
    fprintf('Standard Error: %.6f\n', [std_err_EKF, std_err_UKF, std_err_FEKF, std_err_FEKF_adv](best_idx));
    
    % Improvement analysis
    fprintf('\n=== Improvement Analysis ===\n');
    baseline_error = abs(avr_err_EKF);
    improvements = (baseline_error - abs(errors)) / baseline_error * 100;
    
    for i = 1:length(methods)
        if i == 1
            fprintf('%s: Baseline (0%% improvement)\n', methods{i});
        else
            if improvements(i) > 0
                fprintf('%s: %.2f%% improvement over EKF\n', methods{i}, improvements(i));
            else
                fprintf('%s: %.2f%% degradation compared to EKF\n', methods{i}, -improvements(i));
            end
        end
    end
    
    % Create comparison plot
    figure('Name', 'Method Comparison', 'Position', [100, 100, 1200, 800]);
    
    % Plot 1: Error comparison
    subplot(2,2,1);
    methods_short = {'EKF', 'UKF', 'FEKF-B', 'FEKF-A'};
    bar([abs(avr_err_EKF), abs(avr_err_UKF), abs(avr_err_FEKF), abs(avr_err_FEKF_adv)]);
    set(gca, 'XTickLabel', methods_short);
    ylabel('Average Error');
    title('Average Error Comparison');
    grid on;
    
    % Plot 2: Standard deviation comparison
    subplot(2,2,2);
    bar([std_err_EKF, std_err_UKF, std_err_FEKF, std_err_FEKF_adv]);
    set(gca, 'XTickLabel', methods_short);
    ylabel('Standard Deviation');
    title('Error Standard Deviation Comparison');
    grid on;
    
    % Plot 3: Execution time comparison
    subplot(2,2,3);
    times = [time_original, time_original, time_fekf, time_fekf_adv];
    bar(times);
    set(gca, 'XTickLabel', methods_short);
    ylabel('Execution Time (s)');
    title('Execution Time Comparison');
    grid on;
    
    % Plot 4: Improvement percentage
    subplot(2,2,4);
    bar(improvements);
    set(gca, 'XTickLabel', methods_short);
    ylabel('Improvement (%)');
    title('Improvement over EKF Baseline');
    grid on;
    
    fprintf('\n=== Summary ===\n');
    fprintf('The %s method shows the best performance with %.6f average error.\n', ...
        methods{best_idx}, errors(best_idx));
    
    if best_idx > 1
        fprintf('This represents a %.2f%% improvement over the baseline EKF method.\n', improvements(best_idx));
    end
end