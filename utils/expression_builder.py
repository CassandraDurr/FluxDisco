"""Recursive SymPy expression builder."""

import sympy


class ExpressionBuilder:
    """Helper class to build a SymPy expression from a list of rules."""

    def __init__(
        self,
        action_list: list[str],
        const_offset: int = 0,
        partial_mode: bool = False,
        state_mapping: dict[str, sympy.Symbol] | None = None,
    ):
        """Initialise expression builder.

        Args:
            action_list (list[str]): List of grammar rules (strings).
            const_offset (int, optional): Integer offset for constant numbering. Defaults to 0.
            partial_mode (bool, optional):
                - If True, returns symbol 'M' when rules run out.
                - If False, returns 1.0 (fallback for reward calc).
                - Defaults to False.
            state_mapping (dict[str, sympy.Symbol], optional):
                Mapping of state variable names to SymPy symbols. Defaults to None.
        """
        self.rules = iter(action_list)
        self.const_symbols = []
        self.const_offset = const_offset  # Starting constant index
        self.partial_mode = partial_mode
        self.state_mapping = state_mapping or {}
        self.m_counter = 0  # Counter for non-terminal indexing

    def build_expr(self) -> sympy.Expr:
        """Recursively work through rules iterator to build the expression."""
        try:
            rule = next(self.rules)
        except StopIteration:
            if self.partial_mode:
                # In partial mode, running out of rules means we hit a non-terminal
                # token that hasn't been expanded yet. Return 'M{index}'.
                sym = sympy.Symbol(f"M{self.m_counter}")
                self.m_counter += 1
                return sym
            # Fallback for reward calculation (shouldn't happen)
            return sympy.Float(1.0)

        _, rhs = rule.split(" -> ")

        # Recursive mathematical expressions
        if rhs == "M + M":
            return self.build_expr() + self.build_expr()
        elif rhs == "M - M":
            return self.build_expr() - self.build_expr()
        elif rhs == "M * M":
            return self.build_expr() * self.build_expr()
        elif rhs == "M / M":
            return self.build_expr() / self.build_expr()
        elif rhs == "sqrt(M)":
            # Simple sqrt - add protected sqrt after constant folding
            return sympy.sqrt(self.build_expr())
        # State variables
        elif rhs in self.state_mapping:
            return self.state_mapping[rhs]
        # Constants (terminal rules)
        elif rhs == "C":
            # Create a unique constant symbol across fluxes
            c_idx = len(self.const_symbols) + self.const_offset
            c_sym = sympy.symbols(f"C{c_idx}")
            self.const_symbols.append(c_sym)
            return c_sym
        else:
            # Should not be reached
            return sympy.Float(1.0)


def sympy_expression_builder(
    action_list: list[str],
    state_mapping: dict[str, sympy.Symbol],
    const_offset: int = 0,
    partial_mode: bool = False,
) -> tuple[sympy.Expr, list[sympy.Symbol]]:
    """Convert a list of grammar rules into a SymPy expression for reward calculation.

    Args:
        action_list (list[str]): List of grammar rules (strings).
        state_mapping (dict[str, sympy.Symbol], optional):
            Mapping of state variable names to SymPy symbols. Defaults to None.
        const_offset (int, optional): Integer offset for constant numbering. Defaults to 0.
        partial_mode (bool, optional):
            - If True, returns symbol 'M' when rules run out.
            - If False, returns 1.0 (fallback for reward calc).
            - Defaults to False.

    Returns:
        tuple[sympy.Expr, list[sympy.Symbol]]: SymPy expression, and list of constant symbols used.
    """
    builder = ExpressionBuilder(
        action_list=action_list,
        const_offset=const_offset,
        partial_mode=partial_mode,
        state_mapping=state_mapping,
    )
    # Call the recursive builder
    eq_expr = builder.build_expr()
    # Return the SymPy expression and constants
    return eq_expr, builder.const_symbols
