function d = drdot_du(u, udot, s, sdot)
% This function finds the derivative of range rate with respect
% to the source position.

rdot = (u - s)'*(udot - sdot)/norm(u - s);

d = (udot - sdot)/norm(u - s);

d = d - rdot/norm(u - s) * (u - s)/norm(u - s);
