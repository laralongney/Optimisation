function p = RadicalInverseFun(i, b)
% This function realizes the radical inverse function for input i
% and base b.

bd = b;
p = 0;
while (i > 0)
          a = mod(i, b);
          p = p + a/bd;
          bd = bd * b;
          i = floor(i/b);
end;