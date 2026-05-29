function phi = HammersleySeq(N, b)
% This function generates a N-point Hammersley sequence matrix.

M = numel(b);   % Number of elements in b.

phi = zeros(N, M+1);
phi(:, 1) = (1:N)'/N;   % First coordinate: uniformly spaced.

for m = 1 : M
    for n = 1 : N
        p  = 0;
        i  = n;         % i was never set -- it should start as the sample index n
        bd = b(m);      % bd was never set -- reset the base divisor each sample
        while (i > 0)
            a  = mod(i, b(m));
            p  = p + a/bd;
            bd = bd * b(m);
            i  = floor(i/b(m));
        end
        phi(n, m+1) = p;
    end
end

end