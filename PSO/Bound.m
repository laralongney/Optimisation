function s = Bound(a, p, s, lb, ub)
% This function shrinks the step size s such that lb<= a + s *p <=ub.

if (a + s*p) < lb
    s = (lb - a)/(p + sqrt(eps)); 
    s = max([s, 0]);
end;

if (a + s*p) > ub
    s = (ub - a)/(p + sqrt(eps));
    s = max([s, 0]);
end;