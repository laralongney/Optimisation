function [u, C, iter_count, cost_history, pos_history, gbest_history, theta_out, r1_out] = PSO(fdoa, Q, wavelength, s, sdot, r1_lb, r1_ub, theta_lb, theta_ub, L, B, scaling)
% PSO - Particle Swarm Optimisation for FDOA-only geolocation.
%
% Outputs:
%   u             - Estimated source position in ECEF (3x1, m)
%   C             - Log ML cost at solution
%   iter_count    - Number of iterations run before convergence
%   cost_history  - Global best log cost after each iteration
%   pos_history   - Particle positions at each iteration (2 x N_particles x iter_count)
%   gbest_history - Global best position at each iteration (2 x iter_count)
%   theta_out     - Best theta at convergence (unscaled, rad) for warm-start
%   r1_out        - Best r1 at convergence (unscaled, m) for warm-start

%% --- Scale distances to improve numerical conditioning ---
R     = 6378.137e3 / scaling;
s     = s     / scaling;
r1_lb = r1_lb / scaling;
r1_ub = r1_ub / scaling;

%% --- PSO hyperparameters ---
N_particles = 25;
MaxIter     = 200;
w           = 0.7;
c1          = 1.5;
c2          = 1.5;

%% --- Initialise particle positions randomly across search space ---
theta_range = theta_ub - theta_lb;
r1_range    = r1_ub   - r1_lb;

pos = [theta_lb + rand(1, N_particles) * theta_range;
       r1_lb    + rand(1, N_particles) * r1_range   ];

%% --- Initialise particle velocities ---
vel = [randn(1, N_particles) * theta_range * 0.05;
       randn(1, N_particles) * r1_range    * 0.05];

%% --- Evaluate cost at all initial particle positions ---
costs       = zeros(1, N_particles);
u_particles = zeros(3, N_particles);
for n = 1 : N_particles
    [costs(n), ~, u_particles(:,n)] = MLCost(pos(1,n), pos(2,n), fdoa, Q, ...
        s, sdot, L, B, R, wavelength, scaling);
end

%% --- Initialise personal bests ---
pbest_pos  = pos;
pbest_cost = costs;

%% --- Initialise global best ---
[gbest_cost, gbest_idx] = min(costs);
gbest_pos = pos(:, gbest_idx);
gbest_u   = u_particles(:, gbest_idx);

%% --- Initialise outputs ---
cost_history  = gbest_cost;
iter_count    = 0;

% Pre-allocate position and global best history
pos_history   = zeros(2, N_particles, MaxIter);
gbest_history = zeros(2, MaxIter);

%% --- Main PSO loop ---
for iter = 1 : MaxIter

    for n = 1 : N_particles

        % Velocity update
        r1_rand = rand();
        r2_rand = rand();
        vel(:,n) = w  * vel(:,n)                               ...
                 + c1 * r1_rand * (pbest_pos(:,n) - pos(:,n))  ...
                 + c2 * r2_rand * (gbest_pos      - pos(:,n));

        % Position update
        pos(:,n) = pos(:,n) + vel(:,n);

        % Enforce search bounds (absorbing boundary)
        if pos(1,n) < theta_lb || pos(1,n) > theta_ub
            pos(1,n) = min(max(pos(1,n), theta_lb), theta_ub);
            vel(1,n) = 0;
        end
        if pos(2,n) < r1_lb || pos(2,n) > r1_ub
            pos(2,n) = min(max(pos(2,n), r1_lb), r1_ub);
            vel(2,n) = 0;
        end

        % Evaluate cost at new position
        [c_new, ~, u_new] = MLCost(pos(1,n), pos(2,n), fdoa, Q, ...
            s, sdot, L, B, R, wavelength, scaling);

        % Update personal best
        if c_new < pbest_cost(n)
            pbest_cost(n)  = c_new;
            pbest_pos(:,n) = pos(:,n);
        end

        % Update global best
        if c_new < gbest_cost
            gbest_cost = c_new;
            gbest_pos  = pos(:,n);
            gbest_u    = u_new;
        end

    end % end particle loop

    % Record global best cost and position this iteration
    cost_history            = [cost_history; gbest_cost];
    gbest_history(:, iter)  = gbest_pos;
    iter_count              = iter_count + 1;

    % Snapshot all particle positions for animation
    pos_history(:, :, iter) = pos;

    % All particles have converged to the same location
    pos_spread = max(vecnorm(pos - gbest_pos));
    if pos_spread < 1e-6
        break;
    end

    % Stopping condition: early stagnation detected -- hand off to gradient descent
    if iter > 20
        improvement = cost_history(end-19) - cost_history(end);
        if improvement < 1e-2   % looser threshold -- just hints of stagnation
            break;              % exit PSO early, GD will refine from here
        end
    end

end % end iteration loop

% Trim unused pre-allocated slices
pos_history   = pos_history(:, :, 1:iter_count);
gbest_history = gbest_history(:, 1:iter_count);

%% --- Return results ---
u         = gbest_u * scaling;
C         = gbest_cost;
theta_out = gbest_pos(1);           % scaled theta (rad) -- same scale as input
r1_out    = gbest_pos(2) * scaling; % unscale r1 back to metres for gradient descent
end