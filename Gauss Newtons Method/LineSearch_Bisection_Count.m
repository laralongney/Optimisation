function [a, flag, nfev] = LineSearch_Bisection_Count(phi0, phi0dot, p, theta, theta_lb, theta_ub, r1, r1_lb, r1_ub, fdoa, Q, s, sdot, L, B, R, wavelength, scaling)
% LineSearch_Bisection_Count - Line search with accurate MLCost evaluation counting.
% Replicates LineSearch_Bisection logic exactly, but increments nfev at every
% MLCost call (both in the outer bracketing loop and inside Bisection).

u1    = 1e-4;           % Sufficient decrease factor
u2    = 0.5;            % Sufficient curvature factor
u3    = 1e3/scaling;

phi1    = phi0;
phi1dot = phi0dot;
a1      = 0;

a2_1 = abs(p(1)) / (5*pi/180);
a2_2 = abs(p(2)) / (5e3/scaling);
a2   = 1 / (max(a2_1, a2_2) + sqrt(eps));

sigma = 2;
first = 1;
flag  = 0;
nfev  = 0;   % initialise evaluation counter

while (1)
    a2_1 = Bound(theta, p(1), a2, theta_lb, theta_ub);
    a2_2 = Bound(r1,    p(2), a2, r1_lb,    r1_ub);

    if a2 > min([a2_1, a2_2])
        flag = 1;
    end

    a2 = min([a2_1, a2_2]) * (1 - u3);

    % --- MLCost call 1: bracketing evaluation ---
    [phi2, phi2dot, ~] = MLCost(theta + a2*p(1), r1 + a2*p(2), ...
        fdoa, Q, s, sdot, L, B, R, wavelength, scaling);
    nfev = nfev + 1;
    phi2dot = phi2dot' * p;

    if (phi2 > phi0 + u1*a2*phi0dot) || (~first && (phi2 > phi1))
        % --- Bisection pinpoint ---
        [a, nfev_b] = Bisection_Count(theta, r1, p, u1, u2, phi0, phi0dot, ...
            a1, a2, fdoa, Q, s, sdot, L, B, R, wavelength, scaling);
        nfev = nfev + nfev_b;
        flag = 0;
        break;

    elseif abs(phi2dot) < -u2*phi0dot
        a = a2;
        break;

    elseif phi2dot > 0
        [a, nfev_b] = Bisection_Count(theta, r1, p, u1, u2, phi0, phi0dot, ...
            a1, a2, fdoa, Q, s, sdot, L, B, R, wavelength, scaling);
        nfev = nfev + nfev_b;
        flag = 0;
        break;

    else
        a1      = a2;
        phi1    = phi2;
        phi1dot = phi2dot;

        if flag
            a = a2;
            break;
        else
            a2 = a2 * sigma;
        end
    end

    first = 0;
end
end


function [a, nfev] = Bisection_Count(theta, r1, p, u1, u2, phi0, phi0dot, a1, a2, fdoa, Q, s, sdot, L, B, R, wavelength, scaling)
% Bisection_Count - replicates Bisection.m exactly with MLCost counting.
% Mirrors the original Bisection function — do not change the logic here,
% only add nfev increments at each MLCost call.

nfev   = 0;
MaxBis = 50;

for k = 1 : MaxBis
    am = (a1 + a2) / 2;

    [phim, phimdot, ~] = MLCost(theta + am*p(1), r1 + am*p(2), ...
        fdoa, Q, s, sdot, L, B, R, wavelength, scaling);
    nfev    = nfev + 1;
    phimdot = phimdot' * p;

    if (phim > phi0 + u1*am*phi0dot) || (phim > phi0)   % left side
        a2 = am;
    else
        if abs(phimdot) <= -u2*phi0dot                   % strong Wolfe satisfied
            a = am;
            return;
        elseif phimdot * (a2 - a1) >= 0
            a2 = a1;
        end
        a1 = am;
    end
end

a = am;   % return best estimate if MaxBis reached
end