function [u, C, pos_history_gd, iter_count] = GradDescent(fdoa, Q, wavelength, s, sdot, r1, r1_lb, r1_ub, theta, theta_lb, theta_ub, L, B, scaling)

R     = 6378.137e3 / scaling;
s     = s     / scaling;
r1    = r1    / scaling;
r1_lb = r1_lb / scaling;
r1_ub = r1_ub / scaling;

tolerance = 1e-4;
MaxIter   = 200;

% initialise before loop
pos_history_gd = [theta; r1];
iter_count = 0;
[g, p, u1] = MLCost(theta, r1, fdoa, Q, s, sdot, L, B, R, wavelength, scaling);
g0 = g;

for iter = 1 : MaxIter

    if max(abs(p)) < tolerance
        break;
    end

    d       = p;               % steepest descent direction
    phi0dot = p' * (-d);       % = -p'*p, always negative

    [a, flag, ~] = LineSearch_Bisection_Count(g, phi0dot, -d, ...
        theta, theta_lb, theta_ub, r1, r1_lb, r1_ub, ...
        fdoa, Q, s, sdot, L, B, R, wavelength, scaling);

    theta = theta - a * d(1);
    r1    = r1    - a * d(2);
    pos_history_gd = [pos_history_gd, [theta; r1]];  % append new position
    % inside loop, increment after each update:
    iter_count = iter_count + 1;

    [g, p, u1] = MLCost(theta, r1, fdoa, Q, s, sdot, L, B, R, wavelength, scaling);

    if max(abs(p)) < tolerance;                      break; end
    if abs(g - g0) < tolerance * (1 + abs(g0));     break; end
    if flag == 1;                                    break; end

    g0 = g;
end

u = u1 * scaling;
C = g;
end