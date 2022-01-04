from sys import path

from pathlib import Path
path.append(str(Path().resolve().parents[0]))

from utilities.algorithm.general import check_python_version

check_python_version()

import sys

from algorithm.parameters import params, set_params
from operators.subtree_parse import get_NT_from_str, get_num_from_str, \
    generate_key_and_check, check_snippets_for_solution
from representation.tree import Tree
from utilities.representation.check_methods import generate_codon, \
    check_ind_from_parser
from utilities.stats import trackers


def parse_terminals(target):
    """
    Given a target string, build up a list of terminals which match certain
    portions of the target string.

    :return: A list of terminals in order of appearance in the target string.
    """

    if params['VERBOSE']:
        print("Target:\n", target)

    # Pre-load all terminals and non-terminal rules from the grammar.
    terms, rules = params['BNF_GRAMMAR'].terminals, params['BNF_GRAMMAR'].rules

    # Initialise dict for storing the snippets for compiling a complete
    # solution. The key for each entry is the portion of the target string on
    # which the output matches, along with the root node of the subtree. The
    # value is the subtree itself.
    trackers.snippets = {}

    # Initialise dict for deleted snippets, to ensure they aren't generated
    # again.
    trackers.deleted_snippets = []

    for T in sorted(terms.keys()):
        # Iterate over all Terminals.

        # Find all occurrences of this terminal in the target string.
        occurrences = []
        index = 0
        while index < len(target):
            index = target.find(T, index)
            if index not in occurrences and index != -1:
                occurrences.append(index)
                index += len(T)
            else:
                break

        for idx in occurrences:
            # Check each occurrence of this terminal in the target string.

            for NT in terms[T]:

                if any([[T] == i for i in [[sym['symbol'] for sym in
                                            choice['choice']] for choice in
                                           rules[NT]['choices']]]):
                    # Check if current T is the entire production choice of any
                    # particular rule.

                    # Generate a key for the snippets repository.
                    key = " ".join([str([idx, idx + len(T)]), NT])

                    # Get index of production choice.
                    index = [[sym['symbol'] for sym in choice['choice']] for
                             choice in rules[NT]['choices']].index([T])

                    # Get production choice.
                    choice = rules[NT]['choices'][index]['choice']

                    # Generate a tree for this choice.
                    parent = Tree(NT, None)

                    # Generate a codon for this choice.
                    parent.codon = generate_codon(NT, choice)

                    # Set the snippet key for the parent.
                    parent.snippet = key

                    # Create child for terminal.
                    child = Tree(T, parent)

                    # Add child to parent.
                    parent.children.append(child)

                    # Add snippet to snippets repository.
                    trackers.snippets[key] = parent


def reduce(solution):
    """
    Takes a list of all matching subtrees found in the target string and
    iteratively combines and reduces subtrees to generate larger matching
    subtrees. This process continues until the list of matching subtrees has
    been completely reduced into a target string.

    :param solution: A list of all snippets (i.e. matching subtrees found in
    the target string.
    :return: Nothing.
    """

    # Find all non-terminals in the grammar that can be used to concatenate
    # subtrees to new/larger subtrees.
    reduce_NTs = params['BNF_GRAMMAR'].concat_NTs

    # Pre-load the target string.
    target = params['REVERSE_MAPPING_TARGET']

    for idx, snippet_info in enumerate(solution):
        # Get current snippet.
        snippet = snippet_info[2]

        # Find current snippet info.
        NT = snippet_info[1]

        # Get indexes of the current snippet
        indexes = snippet_info[0]
        start, end = indexes[0], indexes[1]

        # Find if the snippet root (NT) exists anywhere in the
        # reduction NTs.
        if NT in reduce_NTs:

            for reduce in reduce_NTs[NT]:
                # Now we're searching for a specific subset of keys in the
                # snippets dictionary.

                # Generate list of only the desired Non Terminals.
                NTs = reduce[2]

                if len(NTs) == 1:
                    # This choice leads directly to the parent, check if parent
                    # snippet already exists.

                    # Child is current snippet.
                    child = [[snippet, trackers.snippets[snippet]]]

                    # Get key for new snippet.
                    key, start, end = generate_key_and_check(start, end,
                                                             reduce, child)

                    # Create a new node for the solution list.
                    new_entry = [indexes, reduce[1], key]

                    # Insert the current node into the solution.
                    if new_entry not in solution:
                        solution.insert(idx + 1, new_entry)

                else:
                    # Find the index of the snippet root in the current
                    # reduction production choice.
                    NT_locs = [i for i, x in enumerate(NTs) if x[0] == NT]

                    for loc in NT_locs:
                        # We want to check each possible reduction option.

                        # Set where the original snippet starts and ends on
                        # the target string.
                        if loc == 0:
                            # The current snippet is at the start of the
                            # reduction attempt.
                            pre, aft = None, end

                        elif start == 0 and loc != 0:
                            # The current snippet is at the start of the target
                            # string, but we are trying to reduce_trees it with
                            # something before it.
                            break

                        elif end == len(params['TARGET']) and loc != \
                                NT_locs[-1]:
                            # The current snippet is at the end of the target
                            # string, but we are trying to reduce_trees it with
                            # something after it.
                            break

                        elif loc == len(NTs):
                            # The current snippet is at the end of the
                            # reduction attempt.
                            pre, aft = start, None

                        else:
                            # The current snippet is in the middle of the
                            # reduction attempt.
                            pre, aft = start, end

                        alt_cs = list(range(len(NTs)))

                        # Initialise a list of children to be reduced.
                        children = [[] for _ in range(len(NTs))]

                        # Set original snippet into children.
                        children[loc] = [snippet, trackers.snippets[snippet]]

                        curr_idx = solution.index(snippet_info)

                        # Step 1: reduce everything before the current loc.
                        for item in reversed(alt_cs[:loc]):

                            if NTs[item][1] == "T":
                                # This is a terminal, decrement by length of T.

                                # Check output of target string.
                                check = target[pre - len(NTs[item][0]):pre]

                                if check == NTs[item][0]:
                                    # We have a match.

                                    # Generate fake key for snippets dict.
                                    key = str([pre - len(NTs[item][0]), pre])

                                    # Create new tree from this terminal.
                                    T_tree = Tree(check, None)

                                    # Add to children.
                                    children[item] = [key, T_tree]

                                    # Decrement target string index.
                                    pre -= len(NTs[item][0])

                                else:
                                    # No match.
                                    break

                            else:
                                # This is a NT. Check solution list for
                                # matching node.
                                available = [sol for sol in solution[:curr_idx]
                                             if sol[1] == NTs[item][0] and
                                             sol[0][1] == pre]

                                for check in available:
                                    # We have a match.

                                    # Set the correct child in our
                                    # children.
                                    children[item] = [check[2],
                                                      trackers.snippets[
                                                          check[2]]]

                                    # Decrement target string index.
                                    child_len = get_num_from_str(check[2])
                                    pre -= child_len[1] - child_len[0]

                                    break

                        # Step 2: reduce everything after the loc.
                        for i, item in enumerate(alt_cs[loc + 1:]):

                            if NTs[item][1] == "T":
                                # This is a terminal, decrement by length of T.

                                # Check output of target string.
                                check = target[aft: aft + len(NTs[item][0])]

                                if check == NTs[item][0]:
                                    # We have a match.

                                    # Generate fake key for snippets dict.
                                    key = str([aft, aft + len(NTs[item][0])])

                                    # Create new tree from this terminal.
                                    T_tree = Tree(check, None)

                                    # Add to children.
                                    children[item] = [key, T_tree]

                                    # Increment target string index.
                                    aft += len(NTs[item][0])

                                else:
                                    # No match.
                                    break

                            else:
                                # We haven't looked ahead in the string,
                                # we can't add things we don't know yet.
                                break

                        if all([child != [] for child in children]):
                            # We have expanded all children and can collapse
                            # a node.

                            key, pre, aft = generate_key_and_check(pre, aft,
                                                                   reduce,
                                                                   children)

                            # Create a new node for the solution list.
                            new_entry = [[pre, aft], reduce[1], key]

                            # Add the new reduced entry to the solution.
                            if new_entry not in solution:
                                solution.insert(idx + 1, new_entry)


def parse_target_string():
    """
    Takes a list of terminal nodes and iteratively reduces that list until
    the solution has been found.

    :return: The complete parsed solution in the form of a GE individual.
    """

    # Sort snippets keys to generate the initial solution list of terminals.
    solution = sorted([[get_num_from_str(snippet),
                        get_NT_from_str(snippet),
                        snippet] for snippet in trackers.snippets.keys()])

    # Perform reduction on the solution list.
    reduce(solution)

    # Check snippets for solution
    ind = check_snippets_for_solution()

    return ind


def main():
    """
    Run all functions to parse a target string into a GE individual.

    :return: A GE individual.
    """

    # Ensure there is a target to parse.
    if not params['REVERSE_MAPPING_TARGET']:
        s = "scripts.GE_LR_Parser.main\n" \
            "Error: No target string specified for parsing."
        raise Exception(s)

    # Parse the terminals in the target string.
    parse_terminals(params['REVERSE_MAPPING_TARGET'])

    # Iterate over the solution list until the target string is parsed.
    solution = parse_target_string()

    # Check the mapping of the solution and all aspects to ensure it is valid.
    check_ind_from_parser(solution, params['REVERSE_MAPPING_TARGET'])

    return solution


if __name__ == '__main__':
    # Set parameters
    set_params(sys.argv[1:], create_files=False)

    # Print parsed GE genome.
    print("\nGenome:\n", params['SEED_INDIVIDUALS'][0].genome)
