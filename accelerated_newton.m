function [x,x_error,P] = accelerated_newton(func,dfunc,x_initial,convergence_tolerence,Number_of_iterations)

% accelerated_newton                 Accelerated version of Newton's Method
%==========================================================================
%
% USAGE:
%  [x,x_error,P] = accelerated_newton(@func,@dfunc,x_initial,deltax,Number_of_iterations)
%
% DESCRIPTION:
%  This Accelerated version of Newton method for approximating the root of 
%  a univariate function converges faster, with the convergence order being
%  (sqrt(3) + 1) compared with 2 for Newton method. 
%
% Inputs
%  func                   name of the function
%  dfunc                  name of the function's derivative
%  x_initial              Initial estimate of the root
%  convergence_tolerence  convergence tolerance for the root (x)
%  Number_of_iterations   maximum number of iterations
%
%  func & dfunc need to be inputted with an @ prefix. 
%
% OUTPUT:
%   x        solution: the root
%   x_error  the change to x on the last iteration 
%   P        History vector of the values of x during the iterations
%
% AUTHOR: 
%  Trevor McDougall, Simon Wotherspoon & Paul Barker   [ help@teos-10.org ]
%
% VERSION NUMBER: 1 (17th September, 2019)
%
% REFERENCES:
%  McDougall, T.J., S.J. Wotherspoon and P.M. Barker, 2019: An Accelerated
%   version of Newton’s Method with convergence order "sqrt(3) + 1".   
%   Results in Applied Mathematics (submitted). 
%
%==========================================================================

if ~isa(func,'function_handle' ) || ~isa(dfunc,'function_handle' )
    error('the function must be a handle (add an "@" at the begening of the function name)')
end

if ~exist('Number_of_iterations','var')
    Number_of_iterations = 30;
end

if ~exist('convergence_tolerence','var')
    convergence_tolerence = 1e-16;
end

    x = x_initial;
    P = nan(Number_of_iterations,1);
    P(1) = x;
    f = func(x);
    df = dfunc(x);
    delta = f./df;
    x_23 = x - 2.*delta./3;
    df_23 = dfunc(x_23);
    R = df_23./df;
    disc = max(0, 0.75*R - 0.5);
    FF = 0.5 + sqrt(disc);
    x_old = x;
    x = x_old - delta./FF;
    f_old = f;
    df_old = df;
    
for k = 1:Number_of_iterations
    f = func(x);
    df = dfunc(x);
    if df ~= 0
        delta = f./df;
        d2f = (4.*df + 2.*df_old - 6.*(f - f_old)./(x - x_old))./(x - x_old);
        A2_hat = 0.5.*d2f./df;
        disc = max(0, 0.25 - A2_hat.*delta);
        FF = 0.5 + sqrt(disc);
        x_old = x;
        x = x_old - delta./FF;
        df_old = df;
        f_old = f;
    else
        break
    end
    
    P(k+1) = x;
    x_error = abs(x_old - x);
    
    if ((x_error < convergence_tolerence) && (abs(f) < convergence_tolerence))
        P(k+2:Number_of_iterations) = [];
        break
    end
end

end
