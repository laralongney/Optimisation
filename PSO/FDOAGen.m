function fdoa = FDOAGen(u, udot, s1, s1dot, s2, s2dot)
% This function generates the true frequency difference of arrival (FDOA) 
% between sensor pair s1 and s2.

r2dot = (u - s2)'/norm(u - s2) * (udot - s2dot);

r1dot = (u - s1)'/norm(u - s1) * (udot - s1dot);

fdoa = r2dot - r1dot;
