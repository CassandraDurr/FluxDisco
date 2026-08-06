"""Count the number of terminal and non-terminal grammar rules used to build a Sympy expression."""

import math

import sympy


def rules_count(
    expr: sympy.Expr,
    state_vars: set[sympy.Symbol],
    constants_seen: set | None = None,
    division_in_grammar: bool = False,
) -> tuple[int, set | None]:
    """Count the number of grammar rules used to build an expression (post-folding)."""
    # Initialise the constants seen on the first call
    if constants_seen is None:
        constants_seen = set()

    # Separate out terms at +
    terms_list = sympy.Add.make_args(expr)

    # Track the number of rules
    num_rules = 0

    # Count addition operations (number of terms - 1)
    if len(terms_list) > 1:
        num_rules += len(terms_list) - 1

    for term in terms_list:
        if division_in_grammar:
            # Handle fractions
            num, den = sympy.fraction(term)
            if den != 1:
                # Add a rule for the division operator
                num_rules += 1

                # Recursively count the rules in the numerator and denominator
                num_numerator, constants_seen = rules_count(
                    num, state_vars, constants_seen
                )
                num_denominator, constants_seen = rules_count(
                    den, state_vars, constants_seen
                )
                num_rules += num_numerator + num_denominator
                continue

        # Isolate constants and symbols in term
        constant_part, state_part = term.as_independent(*state_vars)

        # Constant multiplied by states (e.g. 3s = s + s + s)
        coeff = 1
        if constant_part.is_Integer and state_part.has(*state_vars):
            coeff = abs(int(constant_part))
            if coeff > 1:
                num_rules += coeff - 1

        # Count variable use of constants (e.g. B0)
        for sym in constant_part.free_symbols:
            if sym not in constants_seen:
                constants_seen.add(sym)
                num_rules += 1

        # Multiplication between constant variable and states (B0 * s)
        if constant_part.free_symbols and state_part.has(*state_vars):
            num_rules += 1

        # Deal with multi-factor states
        if state_part.has(*state_vars):
            # Separate the state into separate factors
            state_factors = state_part.args if state_part.is_Mul else [state_part]

            # Multiplication between state factors (i * s)
            if state_part.is_Mul:
                num_rules += (len(state_part.args) - 1) * coeff

            for state_factor in state_factors:
                # Handle nested structures (sqrt and powers)
                if state_factor.is_Pow:
                    base, exp = state_factor.as_base_exp()

                    # Count rules inside the base
                    base_total, constants_seen = rules_count(
                        base, state_vars, constants_seen
                    )

                    # Split exponent into integer and non-integer parts
                    fractional_part, integer_part = math.modf(float(exp))
                    integer_part = int(integer_part)

                    # Integer powers (s**2 -> s * s)
                    if integer_part > 0:
                        # Variables used
                        num_rules += (integer_part * base_total) * coeff
                        # Operations
                        num_rules += (integer_part - 1) * coeff

                    # Roots (s**0.5 -> sqrt operation)
                    if fractional_part > 0:
                        num_roots = int(1 / (2 * fractional_part))
                        # Variables used
                        num_rules += base_total * coeff
                        # Root operation
                        num_rules += num_roots * coeff

                # Basic state variable (i, s, etc.)
                elif state_factor.is_Symbol:
                    # Variables use
                    num_rules += 1 * coeff

                # Nested additions (like s * (i + 1))
                elif state_factor.is_Add:
                    nested_total, constants_seen = rules_count(
                        state_factor, state_vars, constants_seen
                    )
                    num_rules += nested_total * coeff

    return num_rules, constants_seen
