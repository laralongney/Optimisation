function a = Bisection(theta, r1, p, u1, u2, phi0, phi0dot, a1, a2, fdoa, Q, s, sdot, L, B, R, wavelength, scaling)
% This function searches for a point within the interval [a1, a2] that
% satisfies the strong Wolfe condition.

while(1)
     a = (a1 + a2)/2;
     
     [phi, phidot, ~] = MLCost(theta + a*p(1), r1 + a*p(2), fdoa, Q, s, sdot, L, B, R, wavelength, scaling);
     phidot = phidot' * p;
     
     if (phi > phi0 + u1 * a * phi0dot)
         a2 = a;
         
         continue;
     end;
     
     if (abs(phidot)) < -u2 * phi0dot;
         break;
     elseif phidot > 0
         a2 = a;
         
         continue;
     else
         a1 = a;
         
         continue;
     end;
end;