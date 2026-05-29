function J = grad_computation(theta, r1, L, B, s, sdot, scaling, wavelength)
% grad_computation - Analytically computes the Jacobian J [(M-1) x 2].
%
% Each row of J is the gradient of one predicted FDOA measurement
% with respect to the design variables [theta; r1]:
%   J(m-1, :) = d(FDOA_hat_{m-1}) / d[theta; r1]
%
% Inputs:
%   theta      - Bearing angle (rad)
%   r1         - Source-to-reference-satellite range (scaled)
%   L, B       - Reference satellite longitude/latitude (rad)
%   s          - Satellite positions (scaled, 3xM)
%   sdot       - Satellite velocities (3xM)
%   scaling    - Distance scaling factor
%   wavelength - Signal wavelength (m)

R       = 6378.137e3 / scaling;
M       = size(s, 2);

% --- Reconstruct source position from (theta, r1) ---
s1_norm = norm(s(:,1));
a       = (R^2 - r1^2 + s1_norm^2) / (2*s1_norm);
h_val   = sqrt(max(0, R^2 - a^2) + 1/scaling^2);
T       = ENU2ECEF_Matrix(L, B, 1);
u       = T * [h_val*sin(theta); h_val*cos(theta); a];

% --- du/d[theta; r1] via chain rule through ENU parametrisation ---
dudp_enu = [ h_val*cos(theta),    r1*a/(h_val*s1_norm)*sin(theta);
            -h_val*sin(theta),    r1*a/(h_val*s1_norm)*cos(theta);
             0,                  -r1/s1_norm                      ];
dudp_xyz = T * dudp_enu;

% --- Build Jacobian row by row ---
J = zeros(M-1, 2);

for m = 2 : M
    rm  = u - s(:,m);
    r1v = u - s(:,1);

    % d/du of range-rate for satellite m and reference satellite 1
    drdot_m = sdot(:,m)/norm(rm)  - (rm' *sdot(:,m)) /norm(rm)^3  * rm;
    drdot_1 = sdot(:,1)/norm(r1v) - (r1v'*sdot(:,1)) /norm(r1v)^3 * r1v;

    % Chain rule: d(FDOA)/d[theta,r1] = (1/lambda) * (drdot_m - drdot_1)' * du/dp
    dgdu       = (drdot_m - drdot_1)' / wavelength;
    J(m-1, :)  = dgdu * dudp_xyz;
end

end