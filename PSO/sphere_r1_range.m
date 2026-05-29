function [lb, ub] = sphere_r1_range(s, L, B)
% This function finds the range for the source-reference sensor distance
% under the spheric Earth model.

R = 6378.137e3;

lb = norm(s) - R;

ub = sqrt(s'*s - R^2);

ub1 = sqrt(R^2 + s'*s - 2*norm(s)*R*cos(L));
ub2 = sqrt(R^2 + s'*s - 2*norm(s)*R*cos(B));

ub = min([ub, ub1, ub2]);


