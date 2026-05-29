function [a, flag] = LineSearch_Bisection(phi0, phi0dot, p, theta, theta_lb, theta_ub, r1,  r1_lb, r1_ub, fdoa, Q, s, sdot, L, B, R, wavelength, scaling)
% This function performs the linear search for a given point [theta, r1]'
% along the descent direction p. When a = 0, the cost function value is given in 
% phi0 and the associated derivative is phi0dot.
u1 = 1e-4;                                                        % Sufficient decrease factor.
u2 = 0.5;                                                           % Sufficient curvature factor.
u3 = 1e3/scaling;

phi1 = phi0;
phi1dot = phi0dot;

a1 = 0;

a2_1 = abs(p(1))/(5*pi/180);                         % Interval for theta (=5 degrees).
a2_2 = abs(p(2))/(5e3/scaling);                     % Interval for r1 (=5 km).
a2 = 1/(max(a2_1, a2_2) + sqrt(eps));

sigma = 2; 

first = 1;
flag = 0;                                                            % Boundary flag. 

while (1)
    a2_1 = Bound(theta, p(1), a2, theta_lb, theta_ub);
    a2_2 = Bound(r1, p(2), a2, r1_lb, r1_ub);
    
    if a2 > min([a2_1, a2_2])
       flag = 1;                                                      % At the boundary.
    end
    a2 = min([a2_1, a2_2])*(1-u3);                   % Reduce search interval if boundary crossing occurs.
    
    [phi2, phi2dot, ~] = MLCost(theta + a2*p(1), r1 + a2*p(2), fdoa, Q, s, sdot, L, B, R, wavelength, scaling);
    
    phi2dot = phi2dot' * p;
    
    if (phi2 > phi0 + u1 * a2 * phi0dot) || (~first && (phi2 > phi1))    
        a = Bisection(theta, r1, p, u1, u2, phi0, phi0dot, a1, a2, fdoa, Q, s, sdot, L, B, R, wavelength, scaling);
        flag = 0;                                                     % Still within the boundary.
        
        break;
    end
    
    if abs(phi2dot) < -u2 * phi0dot                 % Strong Wolfe condition.
       a = a2;                                                        % At the boundary or within the boundary.                                                
       break;
    elseif phi2dot > 0
              a = Bisection(theta, r1, p, u1, u2, phi0, phi0dot, a1, a2, fdoa, Q, s, sdot, L, B, R, wavelength, scaling);
              flag = 0;                                               % Still within the boundary.
              
              break;
    else
        a1 = a2;
        phi1 = phi2;
        phi1dot = phi2dot;
        
        if (flag)                                                       % Keep within the boundary.
           a = a2;
           
           break;
        else
           a2 = a2 * sigma;                                    % Expand the interval.
        end
    end
    
    first = 0;
end