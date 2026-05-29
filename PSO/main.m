% FDOA-only Geolocation -- Particle Swarm Optimisation (PSO)
% with Gradient Descent refinement
%
% Produces:
%   Figure 1 -- Geolocation scenario
%   Figure 2 -- RMSE
%   Figure 3 -- Global best cost vs iteration (single example run)
%   Figure 4 -- Distribution of iterations to convergence across runs
%   Figure 5 -- PSO particle animation + GD refinement path

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
sigma_f      = 1;                        % noise standard deviation (Hz)
ensembleRun  = 1000;                     % Monte Carlo runs
gif_filename = 'PSO_GD_16Hz_animation.gif';   % output GIF name

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

%% --- Estimate wavelength ---
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

%% --- Figure 1: Geolocation scenario ---
figure(1);
plot(Lt*rad2deg, Bt*rad2deg, '^k', 'MarkerSize', 12, 'MarkerFaceColor', 'k');
hold on; grid on; box on;
plot(Ls1*rad2deg, Bs1*rad2deg, 'bo', 'MarkerSize', 10, 'MarkerFaceColor', 'b');
plot(Ls2*rad2deg, Bs2*rad2deg, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
plot(Ls3*rad2deg, Bs3*rad2deg, 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
plot(Ls4*rad2deg, Bs4*rad2deg, 'mo', 'MarkerSize', 10, 'MarkerFaceColor', 'm');
xlabel('Longitude (deg)'); ylabel('Latitude (deg)');
title('Geolocation Scenario');
legend('Source', 'Satellite 1 (ref)', 'Satellite 2', 'Satellite 3', 'Satellite 4');
axis([-5, 15, -5, 15]);

%% ================================================================
%  PART A: RMSE -- Monte Carlo simulation
%% ================================================================
err = zeros(ensembleRun, 1);

tic;
for k = 1 : ensembleRun
    noise      = chol(Q)' * randn(M-1, 1);
    FDOA_noisy = fdoa_true + noise;

    [~, ~, ~, ~, ~, ~, theta_out, r1_out] = PSO(FDOA_noisy, Q, lambda_hat, s, sdot, ...
        r1_lb, r1_ub, theta_lb, theta_ub, Ls1, Bs1, scaling);

    % GD refines from PSO global best
    [u_refined, ~] = GradDescent(FDOA_noisy, Q, lambda_hat, s, sdot, ...
        r1_out, r1_lb, r1_ub, theta_out, theta_lb, theta_ub, Ls1, Bs1, scaling);

    err(k) = norm(u_refined - uo)^2;
end
endTime = toc;
fprintf('Total Run Time:   %.4f s\n', endTime);
fprintf('Average Run Time: %.4f s\n', endTime/ensembleRun);

rmse_pso = sqrt(mean(err)) / 1e3;
fprintf('sigma_f = %2.0f Hz --> RMSE PSO+GD = %.3f km\n', sigma_f, rmse_pso);

%% --- Figure 2: RMSE ---
figure(2);
bar(sigma_f, rmse_pso, 'g');
xlabel('Noise std dev \sigma_f (Hz)');
ylabel('RMSE (km)');
title(sprintf('PSO+GD Geolocation RMSE (\\sigma_f = %d Hz)', sigma_f));
grid on; box on;

%% ================================================================
%  PART B: Convergence behaviour -- single example run
%% ================================================================
noise     = chol(Q)' * randn(M-1, 1);
FDOA_demo = fdoa_true + noise;

% PSO global search
[u_demo, C_pso, iter_demo, cost_hist, pos_history, gbest_history, theta_out, r1_out] = ...
    PSO(FDOA_demo, Q, lambda_hat, s, sdot, ...
    r1_lb, r1_ub, theta_lb, theta_ub, Ls1, Bs1, scaling);

% GD refinement from PSO global best -- also capture position history for animation
[u_demo, C_demo, pos_gd] = GradDescent(FDOA_demo, Q, lambda_hat, s, sdot, ...
    r1_out, r1_lb, r1_ub, theta_out, theta_lb, theta_ub, Ls1, Bs1, scaling);

fprintf('\nConvergence demo (sigma_f = %d Hz):\n', sigma_f);
fprintf('PSO iterations:            %d\n',       iter_demo);
fprintf('PSO cost:                  %.4f\n',     C_pso);
fprintf('After GD cost:             %.4f\n',     C_demo);
fprintf('Position error:            %.3f km\n',  norm(u_demo - uo)/1e3);

%% --- Figure 3: Global best cost vs iteration ---
figure(3);
plot(0:length(cost_hist)-1, cost_hist, 'b-', 'LineWidth', 2);
xlabel('Iteration');
ylabel('Global Best Log ML Cost');
title(sprintf('PSO Convergence -- Global Best Cost vs Iteration (\\sigma_f = %d Hz)', sigma_f));
grid on; box on;

%% ================================================================
%  PART C: Iteration count distribution
%% ================================================================
iter_runs = zeros(ensembleRun, 1);
err_runs  = zeros(ensembleRun, 1);

for k = 1 : ensembleRun
    noise  = chol(Q)' * randn(M-1, 1);
    FDOA_k = fdoa_true + noise;

    [~, ~, pso_iters, ~, ~, ~, theta_out, r1_out] = PSO(FDOA_k, Q, lambda_hat, s, sdot, ...
        r1_lb, r1_ub, theta_lb, theta_ub, Ls1, Bs1, scaling);

    [u_k, ~, ~, gd_iters] = GradDescent(FDOA_k, Q, lambda_hat, s, sdot, ...
        r1_out, r1_lb, r1_ub, theta_out, theta_lb, theta_ub, Ls1, Bs1, scaling);

    iter_runs(k) = pso_iters + gd_iters;   % combined total
    err_runs(k)  = norm(u_k - uo) / 1e3;
end
fprintf('\nOver %d runs at sigma_f = %d Hz:\n',           ensembleRun, sigma_f);
fprintf('Mean iterations: %.1f (std: %.1f)\n',           mean(iter_runs), std(iter_runs));
fprintf('Mean position error: %.3f km (std: %.3f km)\n', mean(err_runs),  std(err_runs));

%% --- Figure 4: Histogram of iterations to convergence ---
figure(4);
histogram(iter_runs, 'BinMethod', 'integers', 'FaceColor', [0.2 0.6 0.4]);
xlabel('Iterations to convergence');
ylabel(sprintf('Count (across %d runs)', ensembleRun));
title(sprintf('PSO -- Distribution of Iterations to Convergence (\\sigma_f = %d Hz)', sigma_f));
grid on; box on;

% %% ================================================================
% %  PART D: Particle animation + GD refinement path (Figure 5)
% %% ================================================================
% fprintf('\nBuilding cost function grid for particle animation...\n');
% 
% n_grid    = 60;
% theta_vec = linspace(theta_lb, theta_ub, n_grid);
% r1_vec    = linspace(r1_lb/scaling, r1_ub/scaling, n_grid);
% 
% s_sc = s / scaling;
% R_sc = 6378.137e3 / scaling;
% 
% COST = zeros(n_grid, n_grid);
% for ri = 1 : n_grid
%     for ti = 1 : n_grid
%         try
%             [c_val, ~, ~] = MLCost(theta_vec(ti), r1_vec(ri), ...
%                 FDOA_demo, Q, s_sc, sdot, Ls1, Bs1, R_sc, lambda_hat, scaling);
%             COST(ri, ti) = c_val;
%         catch
%             COST(ri, ti) = NaN;
%         end
%     end
% end
% 
% fprintf('Grid evaluation complete. Starting animation...\n');
% 
% theta_deg_vec = theta_vec * rad2deg;
% N_particles   = size(pos_history, 2);
% 
% %% --- Build Figure 5 ---
% fig5 = figure(5);
% set(fig5, 'Name', 'PSO + GD Animation', 'Color', 'w');
% 
% contourf(theta_deg_vec, r1_vec, COST, 25, 'LineColor', 'none');
% colormap(flipud(hot));
% cb = colorbar;
% cb.Label.String = 'Log ML Cost';
% hold on;
% 
% xlabel('\theta  (degrees)',               'FontSize', 12);
% ylabel('r_1 / scaling  (dimensionless)', 'FontSize', 12);
% h_title = title(sprintf('PSO Particles  --  Iteration: 0 / %d', iter_demo), 'FontSize', 13);
% grid on; box on;
% 
% h_gbest_marker = plot(nan, nan, 'c*',  'MarkerSize', 14, 'LineWidth', 2);
% h_particles    = plot(nan(1, N_particles), nan(1, N_particles), 'wo', ...
%     'MarkerSize', 6, 'MarkerFaceColor', 'w', 'LineWidth', 1);
% h_gd_line      = plot(nan, nan, 'g-',  'LineWidth', 2);
% h_gd_marker    = plot(nan, nan, 'g*',  'MarkerSize', 14, 'LineWidth', 2);
% 
% legend([h_particles, h_gbest_marker, h_gd_line, h_gd_marker], ...
%     {'Particles', 'PSO global best', 'GD path', 'GD current'}, ...
%     'Location', 'northeast', 'TextColor', 'w', 'Color', [0.15 0.15 0.15]);
% 
% drawnow;
% 
% %% --- PSO animation frames ---
% for iter = 1 : iter_demo
% 
%     pos_iter    = pos_history(:, :, iter);
%     theta_deg_p = pos_iter(1, :) * rad2deg;
%     r1_p        = pos_iter(2, :);
% 
%     set(h_particles, 'XData', theta_deg_p, 'YData', r1_p);
%     set(h_gbest_marker, ...
%         'XData', gbest_history(1, iter) * rad2deg, ...
%         'YData', gbest_history(2, iter));
%     set(h_title, 'String', ...
%         sprintf('PSO Particles  --  Iteration: %d / %d', iter, iter_demo));
% 
%     drawnow;
% 
%     frame       = getframe(gcf);
%     img         = frame2im(frame);
%     [imind, cm] = rgb2ind(img, 256);
% 
%     if iter == 1
%         imwrite(imind, cm, gif_filename, 'gif', 'Loopcount', Inf, 'DelayTime', 0.06);
%     else
%         imwrite(imind, cm, gif_filename, 'gif', 'WriteMode', 'append', 'DelayTime', 0.06);
%     end
% end
% 
% %% --- GD refinement animation frames ---
% n_gd_steps = size(pos_gd, 2);
% 
% for iter = 1 : n_gd_steps
% 
%     set(h_gd_line, ...
%         'XData', pos_gd(1, 1:iter) * rad2deg, ...
%         'YData', pos_gd(2, 1:iter));
%     set(h_gd_marker, ...
%         'XData', pos_gd(1, iter) * rad2deg, ...
%         'YData', pos_gd(2, iter));
%     set(h_title, 'String', ...
%         sprintf('Gradient Descent Refinement  --  Step: %d / %d', iter, n_gd_steps));
% 
%     drawnow;
% 
%     frame       = getframe(gcf);
%     img         = frame2im(frame);
%     [imind, cm] = rgb2ind(img, 256);
%     imwrite(imind, cm, gif_filename, 'gif', 'WriteMode', 'append', 'DelayTime', 0.06);
% end
% 
% fprintf('Animation saved to: %s\n', gif_filename);