function [g, p, u] = MLCost(theta, r1, fdoa, Q, s, sdot, L, B, R, wavelength, scaling)
% MLCost - Computes the ML cost function and its gradient for FDOA geolocation.
%
% The cost function is the weighted least squares of FDOA residuals:
%   G = 0.5 * (fdoa - FDOA_pred)' * Q^{-1} * (fdoa - FDOA_pred)
%
% Returns log(G) as the cost (better numerical landscape for line search)
% and the gradient of log(G) w.r.t. [theta; r1].
%
% Inputs:
%   theta      - Bearing angle of source w.r.t. reference satellite (rad)
%   r1         - Distance from source to reference satellite (scaled, m/scaling)
%   fdoa       - Measured FDOA vector (M-1 x 1, Hz)
%   Q          - FDOA noise covariance matrix (M-1 x M-1)
%   s          - Satellite positions (scaled, 3 x M)
%   sdot       - Satellite velocities (3 x M, m/s)
%   L, B       - Reference satellite longitude/latitude (rad)
%   R          - Earth radius (scaled)
%   wavelength - Estimated signal wavelength (m)
%   scaling    - Distance scaling factor
%
% Outputs:
%   g - Log ML cost: log(G)
%   p - Gradient of log(G) w.r.t. [theta; r1] (2x1)
%   u - Reconstructed source position in scaled ECEF (3x1)

% --- Reconstruct source ECEF position from (theta, r1) parametrisation ---
% The source lies on the Earth's surface. Given r1 (distance to satellite 1)
% and theta (bearing angle), we can recover the 3D position via:
%   a = component of source along satellite 1 direction (ENU z-axis)
%   h = component perpendicular to satellite 1 (ENU horizontal plane)
a = (R^2 - r1^2 + s(:,1)'*s(:,1)) / (2*norm(s(:,1)));
h = sqrt(max([0, R^2 - a^2]) + 1/scaling^2);
% Note: 1/scaling^2 is a small regularisation term to prevent h=0 exactly

% Rotate from ENU (local) frame to ECEF (global) frame
T = ENU2ECEF_Matrix(L, B, 1);
u = T * [h*sin(theta); h*cos(theta); a];   % source position in scaled ECEF

% --- Compute predicted FDOAs at the candidate source position u ---
% FDOA between satellite m and reference satellite 1 is the difference
% in range rates (Doppler), divided by wavelength to convert to Hz
M    = size(s, 2);
FDOA = zeros(M-1, 1);
for m = 2 : M
    FDOA(m-1) = FDOAGen(u, zeros(3,1), s(:,1), sdot(:,1), s(:,m), sdot(:,m));
end
FDOA = FDOA / wavelength;   % m/s -> Hz

% --- Raw ML cost G = 0.5 * e' * Q^{-1} * e ---
% e is the residual between measured and predicted FDOAs
% This is the negative log-likelihood under Gaussian noise assumption
g = 0.5 * (fdoa - FDOA)' * inv(Q) * (fdoa - FDOA);

% --- Gradient of G w.r.t. source position u (in ECEF) ---
% Using chain rule: dG/du = -1/lambda * e' * Q^{-1} * d(FDOA)/du
% d(FDOA_m)/du = drdot_m/du - drdot_1/du  (range rate derivatives)
% drdot_du gives d/du of (u-s)'*sdot / |u-s| for one satellite
df_du = zeros(M-1, 3);
for m = 2 : M
    df_du(m-1,:) = drdot_du(u, zeros(3,1), s(:,m), sdot(:,m))';
    df_du(m-1,:) = df_du(m-1,:) - drdot_du(u, zeros(3,1), s(:,1), sdot(:,1))';
end
p = -1/wavelength * (fdoa - FDOA)' * inv(Q) * df_du;   % [1x3] in ECEF

% --- Chain rule: convert gradient from ECEF u to parameters [theta; r1] ---
% du/d[theta; r1] is the Jacobian of the (theta,r1) -> u parametrisation
du_dp = [h*cos(theta),           r1*a/(h*norm(s(:,1)))*sin(theta);
        -h*sin(theta),           r1*a/(h*norm(s(:,1)))*cos(theta);
         0,                     -r1/norm(s(:,1))                  ];
du_dp = T * du_dp;   % rotate to ECEF frame

p = (p * du_dp)';   % [2x1] gradient of G w.r.t. [theta; r1]

% --- Convert to gradient of log(G) ---
% d/dx log(G) = (1/G) * dG/dx
% Use (G + 1/scaling) as denominator floor to prevent blow-up when G -> 0
% near the minimum. This is the lecturer's original numerical stabilisation.
p = p / (g + 1/scaling);   % [2x1] gradient of log(G)

% --- Return log cost (better landscape for line search than raw G) ---
g = log(g);
end