"""Module to protect Sympy expressions with fractional powers and/or fractions.

This could have been done in the expression builder but it complicates constant folding so apply protection after folding.
"""  # noqa: E501

import sympy


def safe_denominator(denominator: sympy.Expr, tolerance: float = 1e-8) -> sympy.Expr:
    """Protect a denominator expression by ensuring it is never too close to zero.

    Ensure the denominator is at least `tolerance` away from zero.
    Also preserve the original sign and ensure smoothness for constant optimisation.

    Args:
        denominator (sympy.Expr): Denominator expression to protect.
        tolerance (float, optional): Minimum distance from zero. Defaults to 1e-8.

    Returns:
        sympy.Expr: Protected denominator expression.
    """
    sign_d = sympy.sign(denominator)
    # sign(0) = 0 -> if sign is 0, make it 1
    # Else, keep the original sign
    safe_sign = sign_d + (1 - sympy.Abs(sign_d))
    # Ensure the magnitude is at least tolerance, but is smooth and differentiable
    return sympy.sqrt(denominator**2 + tolerance**2) * safe_sign


def protect_expr(expr: sympy.Expr) -> sympy.Expr:
    """Recursively protect expressions with fractional powers and/ or fractions."""
    # Infinites
    if expr is sympy.zoo or expr is sympy.oo or expr is sympy.nan:
        print(f"Warning: Expression evaluated to {expr}. Replacing with 1e12.")
        return sympy.Float(1e12)

    # Symbol or number
    if expr.is_Atom:
        return expr

    # Fractions
    num, den = sympy.fraction(expr)
    if den != 1:
        protected_num = protect_expr(num)
        protected_den = protect_expr(den)
        # Protect the denominator
        return protected_num / safe_denominator(protected_den, 1e-8)

    # Powers
    if expr.is_Pow:
        base, exp = expr.as_base_exp()
        protected_base = protect_expr(base)

        # Fractional powers: Ensure base is non-negative
        if not exp.is_Integer:
            protected_base = sympy.Abs(protected_base)

        # Negative powers: Ensure base is non-zero
        if exp.is_negative:
            protected_base = safe_denominator(protected_base, 1e-8)

        return sympy.Pow(protected_base, exp)

    # For all other nodes, process normally
    new_args = [protect_expr(arg) for arg in expr.args]
    return expr.func(*new_args)


def protected_sqrt(expr: sympy.Expr) -> sympy.Expr:
    """Recursively protect expressions with fractional powers."""
    # Symbol or number
    if expr.is_Atom:
        return expr

    # Protect fractional powers
    if expr.is_Pow and not expr.exp.is_Integer:
        # Recursively protect the base first
        protected_base = protected_sqrt(expr.base)
        # Wrap the protected base in Abs() and apply the exponent
        return sympy.Pow(sympy.Abs(protected_base), expr.exp)

    # For all other operations process children normally
    new_args = [protected_sqrt(arg) for arg in expr.args]
    return expr.func(*new_args)
