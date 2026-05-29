function u = sphere_LLA2ECEF(L, B, H)
% This function converts the LLA coordinates into ECEF coordinates under
% spherical Earth model.

R = 6378.137e3;                                              % Earth equatorial radius.

u = [cos(L) * cos(B);
        sin(L)  * cos(B);
        sin(B)];

u = (R+H) * u;