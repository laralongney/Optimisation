function velocity = sphere_ENU2ECEF(L, B, vec)
% This function converts the velocity given in the ENU coordinate system
% into the one given in the ECEF coordinate system.

T = ENU2ECEF_Matrix(L, B, 1);

velocity = T * vec;