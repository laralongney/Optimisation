function  LBH = sphere_ECEF2LLA(u)
% This function converts the ECEF coordinates into LLA coordinates under
% spherical Earth model.

R = 6378.137e3;                                              % Earth equatorial radius.

H = norm(u) - R;                                               % Source altitude.

L = atan2(u(2), u(1));                                       % Source longitude.

B = asin(u(3)/norm(u));                                   % Source latitude.

LBH = [L, B, H]';