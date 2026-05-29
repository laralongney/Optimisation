function T = ENU2ECEF_Matrix(L, B, mode)
% This function calculates the transformation matrix for converting 
% ENU coordinates to ECEF coordinates.

T = [cos(L), -sin(L),  0;
        sin(L),    cos(L), 0;
        0,                   0, 1];

T = T * [cos(B),  0, -sin(B);
              0,          1,         0;
              sin(B),   0,   cos(B)];


if mode == 1
   T = T * [0 0 1;
                 1 0 0;
                 0 1 0];
end;