function [u, C, iter_count, nfev_ls, cost_history, nfev_history] = Newton(fdoa, Q, wavelength, s, sdot, r1, r1_lb, r1_ub, theta, theta_lb, theta_ub, L, B, scaling)
% Newton - FDOA-only geolocation via Newton's method with bisection line search.
%
% Newton's method computes the search direction as:
%   d = H^{-1} * grad
% where H is the approximate Hessian (Gauss-Newton: H = J'*Q^{-1}*J)
% and grad is the gradient of the cost function.
%
% This gives quadratic convergence near the minimum (faster than gradient
% descent) at the cost of needing the Jacobian J at each iteration.
%
% Uses the same MLCost and LineSearch_Bisection as the BFGS implementation,
% so the cost function interface (log cost, scaled gradient) is identical.
%
% Inputs:
%   fdoa           - Noisy FDOA measurements (M-1 x 1, Hz)
%   Q              - FDOA noise covariance (M-1 x M-1)
%   wavelength     - Estimated signal wavelength (m)
%   s              - Satellite positions (3 x M, m)
%   sdot           - Satellite velocities (3 x M, m/s)
%   r1             - Initial source-reference distance (m, unscaled)
%   r1_lb/ub       - Search bounds on r1 (m, unscaled)
%   theta          - Initial bearing angle (rad)
%   theta_lb/ub    - Search bounds on theta (rad)
%   L, B           - Reference satellite longitude/latitude (rad)
%   scaling        - Distance scaling factor (m)
%
% Outputs:
%   u            - Estimated source ECEF position (3x1, m)
%   C            - Log ML cost at solution
%   iter_count   - Number of iterations before convergence
%   nfev_ls      - Total line search function evaluations
%   cost_history - Log cost after each iteration (for convergence plot)
%   nfev_history - Line search function evaluations per iteration

%% --- Scale all distances for numerical conditioning ---
% Dividing by scaling makes r1 ~ O(1) instead of O(1e6),
% keeping it comparable to theta (radians, O(1)) so that
% gradient steps affect both dimensions proportionally
R     = 6378.137e3 / scaling;
s     = s     / scaling;
r1    = r1    / scaling;
r1_lb = r1_lb / scaling;
r1_ub = r1_ub / scaling;

%% --- Algorithm parameters ---
tolerance = 1e-2;   % Convergence threshold on gradient magnitude
MaxIter   = 200;    % Maximum iterations before forced termination

%% --- Evaluate cost and gradient at starting point ---
% MLCost returns: g = log(G), p = gradient of log(G), where G is ML Cost. u1 = ECEF position
[g, p, u1] = MLCost(theta, r1, fdoa, Q, s, sdot, L, B, R, wavelength, scaling);

% Initialise iteration tracking
iter_count   = 0;
nfev_ls      = 0;
cost_history = g;    % record initial cost
nfev_history = [];   % per-iteration line search evaluation counts
g0           = g;    % cost at previous iteration (for stopping check)
x0           = [theta; r1];   % initial scaled values for Option 4 normalisation

% Early exit if already converged at starting point
if max(abs(p)) <= tolerance
    u = u1 * scaling;
    C = g;
    return;
end

%% --- Compute initial Jacobian J [(M-1) x 2] ---
% J(m,:) = d(FDOA_m_predicted)/d[theta, r1]
% Used to build the approximate Hessian H = J'*Q^{-1}*J
J = grad_computation(theta, r1, L, B, s, sdot, scaling, wavelength);

%% --- Precompute Q inverse (constant across iterations) ---
Qinv = inv(Q);

%% --- Main Newton iteration loop ---
for iter = 1 : MaxIter

    %% -- Check gradient convergence before computing direction --
    if max(abs(p)) < tolerance
        break;
    end

    %% -- Build approximate Hessian (Gauss-Newton approximation) --
    % True Hessian of G = J'*Q^{-1}*J + second-order terms
    % Gauss-Newton drops second-order terms (valid near minimum)
    % We then scale by 1/(G + 1/scaling) to get Hessian of log(G),
    % consistent with p which is the gradient of log(G)
    G = exp(g);                                       % recover raw cost from log cost
    H = (J' * Qinv * J) / (G + 1/scaling);           % approximate Hessian of log(G)

    %% -- Compute Newton direction: solve H*d = p --
    % d points toward the minimum of the quadratic approximation
    % If H is singular (ill-conditioned), break (pure Gauss-Newton)
    if rcond(H) < 1e-12
        % d = p;
        break;
    else
        d = H \ p;   % Newton direction: H^{-1} * grad
    end

    %% -- Verify d is a descent direction --
    % The directional derivative of log(G) along -d must be negative
    % for -d to reduce the cost. phi0dot = p'*(-d) should be < 0.
    phi0dot = p' * (-d);
    if phi0dot >= 0
        % H is indefinite, Newton direction is not a descent direction
        % Break for pure Gauss-Newton
        % d       = p;
        % phi0dot = p' * (-p);   % guaranteed negative since p'*p > 0
        break;
    end

    %% -- Line search along direction -d --
    % LineSearch_Bisection finds step length a satisfying strong Wolfe conditions:
    %   Sufficient decrease:  g(x - a*d) <= g(x) + c1*a*phi0dot
    %   Curvature condition:  |d/da g(x - a*d)| <= c2*|phi0dot|
    % This ensures the step is not too large (decrease) or too small (curvature)
    [a, flag, nfev_i] = LineSearch_Bisection_Count(g, phi0dot, -d, ...
        theta, theta_lb, theta_ub, r1, r1_lb, r1_ub, ...
        fdoa, Q, s, sdot, L, B, R, wavelength, scaling);

    nfev_ls      = nfev_ls + nfev_i;        % accumulate total count
    nfev_history = [nfev_history; nfev_i];  % record this iteration's count

    %% -- Save previous values before update (needed for Option 4) --
    theta_prev = theta;
    r1_prev    = r1;

    %% -- Update parameters along Newton direction --
    theta = theta - a * d(1);
    r1    = r1    - a * d(2);

    %% -- Recompute cost, gradient, and Jacobian at new point --
    [g, p, u1] = MLCost(theta, r1, fdoa, Q, s, sdot, L, B, R, wavelength, scaling);
    J = grad_computation(theta, r1, L, B, s, sdot, scaling, wavelength);

    cost_history = [cost_history; g];
    iter_count   = iter_count + 1;

    %% -- Stopping conditions (comment/uncomment as needed) --

    % --- Option 1: Gradient L2 norm ---
    % Stricter - all gradient components must collectively be small
    % if norm(p, 2) < tolerance
    %     break;
    % end

    % --- Option 2: Gradient L-inf norm (default) ---
    % Strongest single-component check
    if max(abs(p)) < tolerance
        break;
    end

    % --- Option 3: Relative change in cost function ---
    % Can converge early if cost stops improving, even if gradient isn't tiny
    if abs(g - g0) < tolerance * (1 + abs(g0))
        break;
    end

    % --- Option 4: Relative change in solution values ---
    % Checks if the actual parameters [theta, r1] have stopped moving
    % x_curr = [theta; r1];
    % x_prev = [theta_prev; r1_prev];
    % if norm(x_curr - x_prev, inf) < tolerance * (1 + norm(x0, inf))
    %     break;
    % end

    % --- Boundary hit: step was clipped to search bound ---
    if flag == 1
        break;
    end

    g0 = g;   % update previous cost for next iteration's stopping check
end

%% --- Unscale and return ---
u = u1 * scaling;   % convert back from scaled to metres
C = g;
end