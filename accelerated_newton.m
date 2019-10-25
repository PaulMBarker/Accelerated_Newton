function [x,x_err,x_h] = accelerated_newton(func,dfunc,x0,conv_tol,n_iter)
 
% accelerated_newton                Accelerated version of Newton's Method
%=========================================================================
%
% USAGE:
%  [x,x_err,x_h] = accelerated_newton(@func,@dfunc,x0,conv_tol,n_iter) 
%
% DESCRIPTION:
%  This accelerated version of Newton’s method for approximating the 
%  root of a univariate function has the order of convergence being 
%  (sqrt(3) + 1) ~ 2.732 compared with the convergence order of 
%  Newton’s method which is 2. 
%
% INPUTS: 
%  func          name of the function 
%  dfunc         name of the function's derivative 
%  x0            initial estimate of the root 
%  conv_tol      convergence tolerance for the root (x)
%  n_iter        maximum number of iterations 
%
%  func & dfunc need to be inputted with an @ prefix. 
%
% OUTPUTS: 
%  x             solution, that is, the root 
%  x_err         the change to x on the last iteration 
%  x_h           history vector of the values of x during the iterations 
%
% AUTHORS: 
%  Trevor McDougall, Simon Wotherspoon & Paul Barker  [ help@teos-10.org ] 
%
% VERSION NUMBER: 1 (25th October, 2019) 
%
% REFERENCES:
%  McDougall, T.J., S.J. Wotherspoon and P.M. Barker, 2019: An Accelerated
%   version of Newton’s Method with convergence order "sqrt(3) + 1".   
%   Results in Applied Mathematics, in press. 
%
%=========================================================================
 
if ~isa(func,'function_handle' ) || ~isa(dfunc,'function_handle' )
  error('function must be a handle (add "@" before the function name)') 
end
 
if ~exist('n_iter','var')
  n_iter = 30; 
end
 
if ~exist('conv_tol','var')
  conv_tol = 1e-15; 
end

x = x0;
x_h = nan(n_iter+1,1); 
x_h(1) = x;
 
for k = 1:n_iter
  f = func(x);
  df = dfunc(x);
  delta = f./df; % -delta is the increment of the regular Newton’s Method.
    
  if k == 1
    df_23 = dfunc(x - 2.*delta./3);    % An extra value of the derivative.
    FF = 0.5 + sqrt(max(0, 0.75.*df_23./df - 0.5));    % FF >= 0.5 always.
    % This value of FF is for our accelerated version of Jarratt’s method.
    % If one prefers this first step to be simply a regular Newton step,
    % then replace the above two lines of code with FF = 1.
  else
    d2f = (4.*df + 2.*df_old - 6.*(f - f_old)./(x - x_old))./(x - x_old);
    A2_hat = 0.5.*d2f./df;
    FF = 0.5 + sqrt(max(0, 0.25 - A2_hat.*delta));     % FF >= 0.5 always.
  end
    
  x_old = x;
  x = x_old - delta./FF;
  x_h(k+1) = x;
  x_err = abs(x_old - x);
    
  if ((x_err < conv_tol) && (abs(f) < conv_tol))
    x_h(k+2:n_iter+1) = [];
    break
  end
  
  df_old = df; f_old = f;
    
  if k == n_iter
    disp('The function did not converge') 
  end
    
end
 
end 
 
