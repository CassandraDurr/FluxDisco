"""Constant folding to simplify expressions prior to optimisation."""

import sympy
from sympy.core.sorting import default_sort_key

from .count_rules import rules_count
from .protected_expr import protect_expr, protected_sqrt


class ConstantFolder:
    """Fold constants in Sympy expressions, simplifying them."""

    def __init__(
        self,
        state_vars: set[sympy.Symbol],
        start_index: int = 0,
        eval_rules: bool = True,
        division_in_grammar: bool = False,
    ):
        """Initialise constant folder."""
        self.state_vars = state_vars
        self.start_index = start_index  # Keep at original value
        self.index = start_index  # Increment as we introduce new K symbols
        self.eval_rules = eval_rules
        self.constant_map = {}
        self.division_in_grammar = division_in_grammar

    def get_or_create_constant(self, c_expr: sympy.Expr) -> sympy.Expr:
        """Get or create a K symbol for a constant expression.

        Args:
            c_expr (sympy.Expr): Sympy expression (constant).

        Returns:
            sympy.Expr: Mapped K symbol, else c_expr itself.
        """
        # If it's purely a number, keep it as is.
        if not c_expr.free_symbols:
            return c_expr

        # If it's already a K symbol, keep it as is.
        if c_expr in self.constant_map.values():
            return c_expr

        # If it's already mapped to a K symbol, return the existing mapping
        if c_expr in self.constant_map:
            return self.constant_map[c_expr]

        # Otherwise, assign a new K symbol
        new_K = sympy.Symbol(f"K{self.index}")
        self.constant_map[c_expr] = new_K
        self.index += 1
        return new_K

    def fold_power(self, base: sympy.Expr, exp: sympy.Expr) -> sympy.Expr:
        """Fold constants inside power expressions.

        Args:
            base (sympy.Expr): Base of power.
            exp (sympy.Expr): Exponent of power.

        Returns:
            sympy.Expr: Folded expression.
        """
        # If the base has no state variables, the entire thing is a constant.
        # No need for folding
        if not base.has(*self.state_vars):
            return base**exp

        # Fold the non-constant base.
        folded_base = self.run_folding(base)

        # Extract multiplicative constants
        if folded_base.is_Mul:
            c_part, s_part = folded_base.as_independent(*self.state_vars)

            if c_part != 1:
                # The base had a constant (e.g., K1 * state_part).
                # Distribute the power: K1**exp * state_part**exp
                new_const_expr = c_part**exp
                final_const = self.get_or_create_constant(new_const_expr)

                # Simplify only the state-dependent product (sqrt(i**2) -> i)
                simplified_s_part = sympy.powdenest(s_part**exp, force=True)
                return final_const * simplified_s_part

        # If it's an Add or something else, simplify the whole thing together.
        return sympy.powdenest(folded_base**exp, force=True)

    def multiplicative_constant_folding(self, expr: sympy.Expr) -> sympy.Expr:
        """Fold multiplicative constants like C1 * C2 * state -> K1 * state.

        Args:
            expr (sympy.Expr): Expression to fold.

        Returns:
            sympy.Expr: Folded expression.
        """
        # Expand/ multiply expression and split terms
        expanded = sympy.expand(expr, self.state_vars, force=True)
        terms = sympy.Add.make_args(expanded)
        folded_terms = []

        for term in terms:
            # Isolate constants in term and symbols in term
            const_part, state_part = term.as_independent(*self.state_vars)

            # Handle internal structure in the state part (like sqrt(i + C0))
            if state_part.has(*self.state_vars) and (
                state_part.is_Mul or state_part.is_Pow
            ):
                if state_part.is_Mul:
                    # Recurse through factors that contain state variables
                    state_part = sympy.Mul(
                        *[
                            (
                                self.run_folding(state_factor)
                                if state_factor.has(*self.state_vars)
                                else state_factor
                            )
                            for state_factor in state_part.args
                        ]
                    )
                elif state_part.is_Pow:
                    base, exp = state_part.as_base_exp()
                    state_part = self.fold_power(base, exp)

            # Map to K symbol using the helper function
            updated_const = self.get_or_create_constant(const_part)

            # Add the new folded term to the list
            folded_terms.append(updated_const * state_part)

        # Recombine the folded terms
        return sum(folded_terms)

    def additive_constant_folding(self, expr: sympy.Expr) -> sympy.Expr:
        """Fold additive constants like C1*state + C2*state -> K1*state.

        Args:
            expr (sympy.Expr): Expression to fold.

        Returns:
            sympy.Expr: Folded expression.
        """
        final_terms = []
        accumulator = {}

        # Separate out terms at +
        terms_list = sympy.Add.make_args(expr)

        for term in terms_list:
            # Split coefficient from state
            coeff, state_part = term.as_independent(*self.state_vars)
            # Add to dictionary
            accumulator[state_part] = accumulator.get(state_part, 0) + coeff

        for state_part, coefficient in accumulator.items():
            updated_coeff = self.get_or_create_constant(coefficient)
            final_terms.append(updated_coeff * state_part)

        return sum(final_terms)

    def run_folding(self, expr: sympy.Expr) -> sympy.Expr:
        """Core folding algorithm that recursively folds constants in the expression.

        Args:
            expr (sympy.Expr): Sympy expression requiring folding.

        Returns:
            sympy.Expr: Folded Sympy expression.
        """
        if self.division_in_grammar:
            num, den = sympy.fraction(expr)
            # Expression is a fraction
            if den != 1:
                # Fold the numerator and denominator independently
                folded_num = self.run_folding(num)
                folded_den = self.run_folding(den)
                if folded_den == 0 or (
                    folded_den.is_Number and abs(folded_den) <= 1e-8
                ):
                    # Prevent division by zero
                    return folded_num / 1e-8
                return folded_num / folded_den

        # Expression is a power
        if expr.is_Pow:
            base, exp = expr.as_base_exp()
            if exp.is_negative and (
                base == 0 or (base.is_Number and abs(base) <= 1e-8)
            ):
                # Prevent zero to negative power
                return sympy.Float(1e12)
            return self.fold_power(base, exp)

        # Multiplicative folding
        multi_folded_expr = self.multiplicative_constant_folding(expr=expr)

        # Additive folding
        add_folded_expr = self.additive_constant_folding(expr=multi_folded_expr)

        return add_folded_expr

    def reindex_constants(
        self, expr: sympy.Expr
    ) -> tuple[sympy.Expr, list[sympy.Symbol]]:
        """Reindex constants from K to B, sorting and ordering the sub-expressions.

        Args:
            expr (sympy.Expr): Sympy expression to reindex.

        Returns:
            tuple[sympy.Expr, list[sympy.Symbol]]: Reindexed expression and list of B constants.
        """
        rename_map = {}
        final_b_index = self.start_index

        # Break the expression into additive terms
        terms = sympy.Add.make_args(expr)

        # Extract (coeff, state_part) so we can sort by state_part
        extracted_terms = [term.as_independent(*self.state_vars) for term in terms]

        # Sort strictly by the canonical mathematical structure of the state part
        extracted_terms.sort(key=lambda x: default_sort_key(x[1]))

        # Assign B-symbols based on the sorted order
        for coeff, state_part in extracted_terms:
            # Recombine to search the whole term, capturing nested K symbols (e.g., inside sqrt)
            full_term = coeff * state_part

            # Extract and sort K symbols found in this specific term
            term_k_syms = sorted(
                [s for s in full_term.free_symbols if s.name.startswith("K")],
                key=lambda s: int(s.name[1:]),
            )

            for k_sym in term_k_syms:
                if k_sym not in rename_map:
                    rename_map[k_sym] = sympy.Symbol(f"B{final_b_index}")
                    final_b_index += 1

        # Apply the mapping
        final_expr_reindexed = expr.subs(rename_map)

        # Sort B-consts to look nice in the list [B0, B1, B2]
        final_consts = sorted(  # noqa: C414
            list(rename_map.values()), key=lambda s: int(s.name[1:])
        )

        return final_expr_reindexed, final_consts

    def fold_and_protect(
        self, expr: sympy.Expr
    ) -> tuple[sympy.Expr, int, list[sympy.Symbol], int | None]:
        """Fold and protect Sympy expression.

        Args:
            expr (sympy.Expr): Sympy expression to fold and protect.

        Returns:
            tuple[sympy.Expr, int, list[sympy.Symbol], int | None]:
            - Folded and protected expression.
            - Next constant index after folding.
            - List of folded B constants.
            - Rule count in folded expression (if eval_rules), else None.
        """
        if expr is sympy.zoo or expr is sympy.oo or expr is sympy.nan:
            print(f"Warning: Original expression evaluated to {expr}")

        # Recursive folding
        folded_expr = self.run_folding(expr)

        if (
            folded_expr is sympy.zoo
            or folded_expr is sympy.oo
            or folded_expr is sympy.nan
        ):
            print(f"Warning: Folded expression evaluated to {folded_expr}")

        # Drop imaginary part if it arises
        folded_expr = folded_expr.subs(sympy.I, 1)

        if (
            folded_expr is sympy.zoo
            or folded_expr is sympy.oo
            or folded_expr is sympy.nan
        ):
            print(
                f"Warning: Folded expression post-imaginary evaluated to {folded_expr}"
            )

        # Re-index expressions
        reindexed_expr, final_constants = self.reindex_constants(folded_expr)

        if (
            reindexed_expr is sympy.zoo
            or reindexed_expr is sympy.oo
            or reindexed_expr is sympy.nan
        ):
            print(f"Warning: Re-indexed expression evaluated to {reindexed_expr}")

        # Calculate the final B index (start index + number of unique B constants introduced)
        final_b_index = self.start_index + len(final_constants)

        # Count rules before protections
        if self.eval_rules:
            rules, _ = rules_count(
                expr=reindexed_expr,
                state_vars=self.state_vars,
                division_in_grammar=self.division_in_grammar,
            )
        else:
            rules = None

        # Protect the expression when it contains roots, fractional powers, or fractions
        if self.division_in_grammar:
            protected_expr = protect_expr(reindexed_expr)
        else:
            protected_expr = protected_sqrt(reindexed_expr)

        return (
            protected_expr,
            final_b_index,  # Next constant index after folding
            final_constants,
            rules,
        )


# --- Helper functions to perform constant folding ---
def collect_partial_state_vars(expr: sympy.Expr, base_state_vars: set):
    """
    Collect state variables for partial expressions.

    State variables:
      - base state variables: base system state variables (e.g., s, i, r)
      - any M{int} symbols present in the expression
    """
    # Add all M{int} symbols referenced in expr
    m_syms = {
        sym
        for sym in expr.free_symbols
        if sym.name.startswith("M") and sym.name[1:].isdigit()
    }
    return base_state_vars | m_syms


def constant_folding(
    expr: sympy.Expr,
    state_vars: set[sympy.Symbol],
    start_index: int = 0,
    eval_rules: bool = True,
    division_in_grammar: bool = False,
) -> tuple[sympy.Expr, int, list[sympy.Symbol], int | None]:
    """
    Constant folding for terminal/ complete expressions.

    Args:
        expr (sympy.Expr): The expression to fold.
        state_vars (set[sympy.Symbol]): Set of state variables.
        start_index (int): Starting index for introduced K/B symbols.
        eval_rules (bool): Whether to count rules in the folded expression.
        division_in_grammar (bool): Whether division is allowed in the grammar.

    Returns:
        tuple: (folded_expression, next_constant_index, folded_B_constants, rules_count or None)
    """
    folder = ConstantFolder(
        state_vars=state_vars,
        start_index=start_index,
        eval_rules=eval_rules,
        division_in_grammar=division_in_grammar,
    )
    return folder.fold_and_protect(expr)


def constant_folding_partial(
    expr: sympy.Expr,
    state_vars_base: set,
    start_index: int = 0,
    eval_rules: bool = True,
    division_in_grammar: bool = False,
) -> tuple[sympy.Expr, int, list[sympy.Symbol], int | None]:
    """
    Constant folding for partial expressions that contain M0, M1, ... tokens.

    Args:
        expr (sympy.Expr): The partial expression to fold.
        state_vars_base (set[sympy.Symbol]): Base state variables.
        start_index (int): Starting index for introduced K/B symbols.
        eval_rules (bool): Whether to count rules in the folded expression.
        division_in_grammar (bool): Whether division is allowed in the grammar.

    Returns:
        tuple: (folded_expression, next_constant_index, folded_B_constants, rules_count or None)
    """
    state_vars = collect_partial_state_vars(expr, state_vars_base)
    folder = ConstantFolder(
        state_vars=state_vars,
        start_index=start_index,
        eval_rules=eval_rules,
        division_in_grammar=division_in_grammar,
    )
    return folder.fold_and_protect(expr)
