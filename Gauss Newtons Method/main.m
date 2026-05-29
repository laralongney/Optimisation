% FDOA-only Geolocation -- Newton's Method
clc; close all; clear;

c       = 299792458;
deg2rad = pi/180;
rad2deg = 180/pi;
f0      = 1e9;
scaling = 1000e3;

wavelength = c/f0;

%% ================================================================
%  GLOBAL SETTINGS -- change these only
%% ================================================================
sigma_f     = 1;    % noise standard deviation (Hz)
ensembleRun = 10000;  % Monte Carlo runs

%% --- True source position ---
Lt = 10*deg2rad;  Bt = 5*deg2rad;  Ht = 0;
uo = sphere_LLA2ECEF(Lt, Bt, Ht);

%% --- Satellite positions (ECEF, metres) ---
s1 = [7378.1e3;    0;      0    ];
s2 = [7377.5e3;  100e3;    0    ];
s3 = [7377.5e3; -100e3;    0    ];
s4 = [7377.5e3;    0;    100e3 ];

%% --- Satellite velocities (ECEF, m/s) ---
s1dot = [ 0.0001e3; 4.4995e3; 5.3623e3];
s2dot = [-0.0671e3; 4.9493e3; 4.9497e3];
s3dot = [ 0.0610e3; 4.4991e3; 5.3623e3];
s4dot = [-0.0777e3; 4.0150e3; 5.7335e3];

s    = [s1, s2, s3, s4];
sdot = [s1dot, s2dot, s3dot, s4dot];
M    = size(s, 2);

%% --- Convert ECEF satellite positions to LLA for plotting and MLCost ---
LBH1 = sphere_ECEF2LLA(s1);  Ls1 = LBH1(1);  Bs1 = LBH1(2);
LBH2 = sphere_ECEF2LLA(s2);  Ls2 = LBH2(1);  Bs2 = LBH2(2);
LBH3 = sphere_ECEF2LLA(s3);  Ls3 = LBH3(1);  Bs3 = LBH3(2);
LBH4 = sphere_ECEF2LLA(s4);  Ls4 = LBH4(1);  Bs4 = LBH4(2);

%% --- Estimate wavelength from raw frequency measurements ---
sigma_est  = 16;
f_meas     = f0*ones(M,1) + sigma_est*randn(M,1);
f0_hat     = mean(f_meas);
lambda_hat = c / f0_hat;
fprintf('True wavelength:      %.6f m\n', wavelength);
fprintf('Estimated wavelength: %.6f m\n', lambda_hat);
fprintf('Relative error:       %.4e\n\n', abs(lambda_hat-wavelength)/wavelength);

%% --- True noise-free FDOAs ---
fdoa_true = zeros(M-1, 1);
for m = 2:M
    fdoa_true(m-1) = FDOAGen(uo, zeros(3,1), s(:,1), sdot(:,1), s(:,m), sdot(:,m));
end
fdoa_true = fdoa_true / wavelength;

%% --- Noise covariance ---
Q = sigma_f^2 * (eye(M-1) + ones(M-1)) / 2;

%% --- Search bounds ---
L_range = 40*deg2rad;  B_range = 40*deg2rad;
[r1_lb, r1_ub] = sphere_r1_range(s(:,1), L_range, B_range);
theta_lb = -pi/2;  theta_ub = pi/2;

%% --- Multistart (Hammersley) ---
N_starts    = 20;
b           = 2;
delta       = 0.05;
delta_r1    = (r1_ub - r1_lb) * delta;
delta_theta = (theta_ub - theta_lb) * delta;
% phi         = HammersleySeq(N_starts, b);
% r1_0    = r1_lb + delta_r1    + phi(:,1)*(r1_ub - r1_lb - 2*delta_r1);
% theta_0 = theta_lb + delta_theta + phi(:,2)*(theta_ub - theta_lb - 2*delta_theta);

% --- Multistart (Random) -- uncomment to use instead of Hammersley ---
r1_0    = r1_lb + delta_r1    + rand(N_starts, 1) * (r1_ub - r1_lb - 2*delta_r1);
theta_0 = theta_lb + delta_theta + rand(N_starts, 1) * (theta_ub - theta_lb - 2*delta_theta);

%% --- Figure 1: Scenario ---
figure(1);
plot(Lt*rad2deg, Bt*rad2deg, '^k', 'MarkerSize', 12, 'MarkerFaceColor','k');
hold on; grid on; box on;
plot(Ls1*rad2deg, Bs1*rad2deg, 'bo', 'MarkerSize', 10, 'MarkerFaceColor', 'b');
plot(Ls2*rad2deg, Bs2*rad2deg, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
plot(Ls3*rad2deg, Bs3*rad2deg, 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
plot(Ls4*rad2deg, Bs4*rad2deg, 'mo', 'MarkerSize', 10, 'MarkerFaceColor', 'm');
xlabel('Longitude (deg)'); ylabel('Latitude (deg)');
title('Geolocation Scenario');
legend('Source','Sat 1','Sat 2','Sat 3','Sat 4');
axis([-5,15,-5,15]);

%% --- Figure 6: Initial starting points ---
figure(6);
scatter(theta_0 * rad2deg, r1_0 / 1e3, 60, 'b', 'filled');
grid on; box on;
xlabel('Theta (deg)');
ylabel('r1 (km)');
title('Initial Starting Points in Search Space');
xlim([theta_lb * rad2deg, theta_ub * rad2deg]);
ylim([r1_lb / 1e3, r1_ub / 1e3]);

%% ================================================================
%  PART A: RMSE -- Monte Carlo simulation
%% ================================================================
err = zeros(ensembleRun, 1);

tic;
for k = 1 : ensembleRun
    noise      = chol(Q)' * randn(M-1, 1);
    FDOA_noisy = fdoa_true + noise;

    u_ms = zeros(3, N_starts);
    C_ms = zeros(N_starts, 1);

    for n = 1 : N_starts
        [u_ms(:,n), C_ms(n), ~, ~, ~, ~] = Newton(FDOA_noisy, Q, lambda_hat, ...
            s, sdot, r1_0(n), r1_lb, r1_ub, theta_0(n), theta_lb, theta_ub, ...
            Ls1, Bs1, scaling);
    end

    [~, idx] = min(C_ms);
    err(k)   = norm(u_ms(:,idx) - uo)^2;
end
runtime = toc;

rmse_newton = sqrt(mean(err)) / 1e3;
fprintf('sigma_f = %2.0f Hz --> RMSE = %.3f km\n', sigma_f, rmse_newton);
fprintf('Total time: %.4f s  |  Avg per run: %.4f s\n\n', runtime, runtime/ensembleRun);

%% --- Figure 2: RMSE ---
figure(2);
bar(sigma_f, rmse_newton, 'b');
xlabel('Noise std dev \sigma_f (Hz)');
ylabel('RMSE (km)');
title(sprintf('Gauss-Newton Geolocation RMSE (\\sigma_f = %d Hz)', sigma_f));
grid on; box on;

%% ================================================================
%  PART B: Convergence at chosen sigma_f
%% ================================================================
noise     = chol(Q)' * randn(M-1, 1);
FDOA_demo = fdoa_true + noise;

u_demo    = zeros(3, N_starts);
C_demo    = zeros(N_starts, 1);
iter_demo = zeros(N_starts, 1);
nfev_demo = zeros(N_starts, 1);
cost_hist = cell(N_starts, 1);
nfev_hist = cell(N_starts, 1);

for n = 1 : N_starts
    [u_demo(:,n), C_demo(n), iter_demo(n), nfev_demo(n), cost_hist{n}, nfev_hist{n}] = ...
        Newton(FDOA_demo, Q, lambda_hat, s, sdot, ...
        r1_0(n), r1_lb, r1_ub, theta_0(n), theta_lb, theta_ub, Ls1, Bs1, scaling);
end

% Debug printout
fprintf('Run-by-run summary:\n');
for n = 1 : N_starts
    fprintf('  Run %2d: C = %8.4f, iters = %d\n', n, C_demo(n), iter_demo(n));
end

[~, best] = min(C_demo);
fprintf('\nBest run: %d (C = %.4f)\n\n', best, C_demo(best));

%% --- Figure 3: Cost vs iteration ---
figure(3); hold on; grid on; box on;
h_other = [];
for n = 1 : N_starts
    if n ~= best
        h_other = plot(0:length(cost_hist{n})-1, cost_hist{n}, ...
                       'Color', [0.7 0.7 0.7], 'LineWidth', 0.8);
    end
end
h_best = plot(0:length(cost_hist{best})-1, cost_hist{best}, 'b-', 'LineWidth', 2);
xlabel('Iteration'); ylabel('Log ML Cost');
title(sprintf('Convergence: Cost vs Iteration (\\sigma_f = %d Hz)', sigma_f));
legend([h_other, h_best], 'Other starts', 'Best start', 'Location', 'northeast');
xlim([0, max(cellfun(@length, cost_hist)) - 1]);

%% --- Figure 4: Iterations histogram ---
figure(4);
histogram(iter_demo, 'BinMethod', 'integers', 'FaceColor', [0.2 0.5 0.8]);
xlabel('Iterations to convergence'); ylabel('Count');
title(sprintf('Newton''s Method -- Iterations to Convergence (\\sigma_f = %d Hz)', sigma_f));
grid on; box on;

%% --- Figure 5: Per-iteration line search evaluations ---
figure(5);
all_nfev_per_iter = [];
for n = 1 : N_starts
    all_nfev_per_iter = [all_nfev_per_iter; nfev_hist{n}(:)];
end
histogram(all_nfev_per_iter, 'BinMethod', 'integers', 'FaceColor', [0.8 0.4 0.2]);
xlabel('Line search evaluations (per iteration)'); ylabel('Count');
title(sprintf('Newton''s Method -- Line Search Evals per Iteration (\\sigma_f = %d Hz)', sigma_f));
grid on; box on;