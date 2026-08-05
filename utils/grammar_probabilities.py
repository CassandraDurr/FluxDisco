"""Helper function to initialise the valid grammar rules and their corresponding probabilities."""


def initialise_grammar_probabilities_util(
    grammar: dict,
    flux_priors: dict,
    terminal_rules_for_M: list[str],
    num_fluxes: int,
    state_keys: list[str],
):
    """
    Calculate normalised probabilities for the flux grammar rules based on priors.

    Valid M rules used in rollout & getting untried actions, valid terminal M rules used in force-completion.
    Valid M terminal state rules used in rollout to ensure branches have at least one state.

    Define:
        - valid_M_rules: rules allowed per flux
        - M_rule_probs: probabilities of rules allowed per flux
        - valid_terminal_M_rules: terminal rules allowed per flux
        - terminal_M_rule_probs: probabilities of terminal rules allowed per flux
        - valid_state_terminal_M_rules: terminal rules allowed per flux containing state values
        - terminal_M_rule_probs: probabilities of terminal rules allowed per flux containing state values
    """  # noqa: E501
    valid_M_rules = {}
    M_rule_probs = {}
    valid_terminal_M_rules = {}
    terminal_M_rule_probs = {}
    valid_state_terminal_M_rules = {}
    valid_state_terminal_M_rule_probs = {}

    # Get valid terminal M rules with states
    required_chars = set(state_keys)

    for flux in range(num_fluxes):
        # Check if we have priors for this specific flux
        if flux in flux_priors:
            flux_specific_priors: dict = flux_priors[flux]

            # Filter only rules with > 0 prior
            valid_rules_map = {
                key: val for key, val in flux_specific_priors.items() if val > 0
            }

            if not valid_rules_map:
                raise ValueError("All flux priors are zero.")
            else:
                # Normalise all valid rules
                total_score = sum(valid_rules_map.values())
                rules = list(valid_rules_map.keys())
                probs = [valid_rules_map[rule] / total_score for rule in rules]

                valid_M_rules[flux] = rules
                M_rule_probs[flux] = probs

                # Normalise terminal rules (subset of rules for force completion)
                term_map = {
                    key: val
                    for key, val in valid_rules_map.items()
                    if key in terminal_rules_for_M
                }

                if not term_map:
                    raise ValueError("All flux priors for terminal rules are zero.")
                else:
                    total_term_score = sum(term_map.values())
                    term_rules = list(term_map.keys())
                    term_probs = [
                        term_map[rule] / total_term_score for rule in term_rules
                    ]

                    valid_terminal_M_rules[flux] = term_rules
                    terminal_M_rule_probs[flux] = term_probs

        else:
            # No priors provided for this flux -> assume random selection.
            # All M rules
            rules = grammar["M"]
            probs = [1.0 / len(rules)] * len(rules)
            valid_M_rules[flux] = rules
            M_rule_probs[flux] = probs

            # Terminal M rules
            term_rules = terminal_rules_for_M
            term_probs = [1.0 / len(term_rules)] * len(term_rules)
            valid_terminal_M_rules[flux] = term_rules
            terminal_M_rule_probs[flux] = term_probs

        # Filter the valid terminal rules and get their corresponding probabilities
        filtered_rules_with_probs = [
            (rule, prob)
            for rule, prob in zip(
                valid_terminal_M_rules[flux], terminal_M_rule_probs[flux]
            )
            if any(char in rule for char in required_chars)
        ]
        filtered_rules = [item[0] for item in filtered_rules_with_probs]
        filtered_probs = [item[1] for item in filtered_rules_with_probs]

        # Normalise the filtered probabilities
        sum_of_probs = sum(filtered_probs)
        normalised_filtered_probs = [p / sum_of_probs for p in filtered_probs]

        valid_state_terminal_M_rules[flux] = filtered_rules
        valid_state_terminal_M_rule_probs[flux] = normalised_filtered_probs

    return (
        valid_M_rules,
        M_rule_probs,
        valid_terminal_M_rules,
        terminal_M_rule_probs,
        valid_state_terminal_M_rules,
        valid_state_terminal_M_rule_probs,
    )
