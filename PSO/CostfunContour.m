function p = CostfunContour(r1_lb, r1_ub, theta_lb, theta_ub, s, sdot, L, B, fdoa, wavelength, Q, scaling)
% This function plots the contour lines of the logarithm of the maximum likelihood (ML)
% cost function in polar and LLA coordinates systems.
deg2rad = pi/180;                                           % Degree to radian conversion.
rad2deg = 180/pi;                                           % Radian to degree conversion.

R = 6378.137e3;

R = R/scaling;                                                   % Scaled Earth radius.
s  = s/scaling;                                                   % Scaled sensor positions.
r1_lb = r1_lb/scaling;
r1_ub = r1_ub/scaling;

r1 = r1_lb : 0.02 : r1_ub;
theta = theta_lb : 1*deg2rad: theta_ub;

M = numel(r1);
N = numel(theta);

cost = zeros(M*N, 1);                                           
u_ECEF = zeros(3, M*N);
u_LLA = zeros(3, M*N);

T = ENU2ECEF_Matrix(L, B, 1);

j = 0;
for m = 1 : M
      a = (R^2 - r1(m)^2 + s(:, 1)'*s(:, 1))/(2*norm(s(:, 1)));
      h = sqrt(R^2 - a^2 + 1/scaling^2);
      
      for n = 1 : N          
            j = j + 1;
            u = [h*sin(theta(n)), h*cos(theta(n)), a]';
           
            u = T * u;
            u_ECEF(:, j) = u * scaling;
            u_LLA(:, j) = sphere_ECEF2LLA(u * scaling);
            
            K = size(s, 2);
            FDOA = zeros(K-1, 1);
            
            for i = 2 : K
                  FDOA(i - 1) = FDOAGen(u, zeros(3,1), s(:, 1), sdot(:, 1), s(:, i), sdot(:, i));
            end;
            FDOA = FDOA/wavelength;
            
            cost(j) = log((FDOA - fdoa)' * inv(Q) * (FDOA - fdoa)/2);
     end;
end;

figure(3); 
plot3(u_LLA(1, :)*rad2deg, u_LLA(2, :)*rad2deg, cost, '.-');
grid on;
xlabel('Longitude (Degree)');
ylabel('Latitude (Degree)');

figure(4);
[X, Y] = meshgrid(theta*rad2deg, r1);
cost = reshape(cost, N, M);
Z = cost';

contour(X, Y, Z, 9, 'ShowText', 'on');
xlabel('\theta (Degree)');
ylabel('r_1 (1000km)');
grid on;
title(' Cost function contour')

p = 1;