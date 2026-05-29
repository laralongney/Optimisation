function [g, p, u] = MLCost(theta, r1, fdoa, Q, s, sdot, L, B, R, wavelength, scaling)

a = (R^2 - r1^2 + s(:,1)'*s(:,1)) / (2*norm(s(:,1)));
h = sqrt(max([0, R^2 - a^2]) + 1/scaling^2);

T = ENU2ECEF_Matrix(L, B, 1);
u = T * [h*sin(theta); h*cos(theta); a];

M = size(s, 2);
FDOA = zeros(M-1, 1);
for m = 2 : M
    FDOA(m-1) = FDOAGen(u, zeros(3,1), s(:,1), sdot(:,1), s(:,m), sdot(:,m));
end
FDOA = FDOA / wavelength;

g = 0.5 * (fdoa - FDOA)' * inv(Q) * (fdoa - FDOA);

df_du = zeros(M-1, 3);
for m = 2 : M
    df_du(m-1,:) = drdot_du(u, zeros(3,1), s(:,m), sdot(:,m))';
    df_du(m-1,:) = df_du(m-1,:) - drdot_du(u, zeros(3,1), s(:,1), sdot(:,1))';
end

p = -1/wavelength * (fdoa - FDOA)' * inv(Q) * df_du;

du_dp = [h*cos(theta),           r1*a/(h*norm(s(:,1)))*sin(theta);
        -h*sin(theta),           r1*a/(h*norm(s(:,1)))*cos(theta);
         0,                     -r1/norm(s(:,1))];
du_dp = T * du_dp;

p = (p * du_dp)';
p = p / (g + 1/scaling);   % gradient of log(g)

g = log(g);                 % log cost
end