from math import floor
from re import DOTALL, MULTILINE, finditer, match
from sys import maxsize
from itertools import chain, combinations
import pandas as pd

from algorithm.parameters import params

def powerset(iterable, minlen=0, maxlen=None):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    if maxlen is None: maxlen = len(s)
    return chain.from_iterable(combinations(s, r)
                               for r in range(minlen, maxlen+1))


class Grammar(object):
    """
    Parser for Backus-Naur Form (BNF) Context-Free Grammars.
    """

    def __init__(self, file_name):
        """
        Initialises an instance of the grammar class. This instance is used
        to parse a given file_name grammar.

        :param file_name: A specified BNF grammar file.
        """

        if file_name.endswith("pybnf"):
            # Use python filter for parsing grammar output as grammar output
            # contains indented python code.
            self.python_mode = True

        else:
            # No need to filter/interpret grammar output, individual
            # phenotypes can be evaluated as normal.
            self.python_mode = False

        # Initialise empty dict for all production rules in the grammar.
        # Initialise empty dict of permutations of solutions possible at
        # each derivation tree depth.
        self.rules, self.permutations = {}, {}

        # Initialise dicts for terminals and non terminals, set params.
        self.non_terminals, self.terminals = {}, {}
        self.start_rule, self.codon_size = None, params['CODON_SIZE']
        self.min_path, self.max_arity, self.min_ramp = None, None, None

        # Set regular expressions for parsing BNF grammar.
        self.ruleregex = '(?P<rulename><\S+>)\s*::=\s*(?P<production>(?:(?=\#)\#[^\r\n]*|(?!<\S+>\s*::=).+?)+)'
        self.productionregex = '(?=\#)(?:\#.*$)|(?!\#)\s*(?P<production>(?:[^\'\"\|\#]+|\'.*?\'|".*?")+)'
        self.productionpartsregex = '\ *([\r\n]+)\ *|([^\'"<\r\n]+)|\'(.*?)\'|"(.*?)"|(?P<subrule><[^>|\s]+>)|([<]+)'

        # to speed up the recursion step
        self.recursion_cache = {}

        # Read in BNF grammar, set production rules, terminals and
        # non-terminals.
        self.read_bnf_file(file_name)

        # Check the minimum depths of all non-terminals in the grammar.
        self.check_depths()

        # Check which non-terminals are recursive.
        self.check_recursion(self.start_rule["symbol"], [])

        # Set the minimum path and maximum arity of the grammar.
        self.set_arity()

        # Generate lists of recursive production choices and shortest
        # terminating path production choices for each NT in the grammar.
        # Enables faster tree operations.
        self.set_grammar_properties()

        # Calculate the total number of derivation tree permutations and
        # combinations that can be created by a grammar at a range of depths.
        self.check_permutations()

        if params['MIN_INIT_TREE_DEPTH']:
            # Set the minimum ramping tree depth from the command line.
            self.min_ramp = params['MIN_INIT_TREE_DEPTH']

        elif hasattr(params['INITIALISATION'], "ramping"):
            # Set the minimum depth at which ramping can start where we can
            # have unique solutions (no duplicates).
            self.get_min_ramp_depth()

        if params['REVERSE_MAPPING_TARGET'] or params['TARGET_SEED_FOLDER']:
            # Initialise dicts for reverse-mapping GE individuals.
            self.concat_NTs, self.climb_NTs = {}, {}

            # Find production choices which can be used to concatenate
            # subtrees.
            self.find_concatenation_NTs()

    def read_bnf_file(self, file_name):
        """
        Read a grammar file in BNF format. Parses the grammar and saves a
        dict of all production rules and their possible choices.

        :param file_name: A specified BNF grammar file.
        :return: Nothing.
        """

        with open(file_name, 'r') as bnf:
            # Read the whole grammar file.
            content = bnf.read()

            for rule in finditer(self.ruleregex, content, DOTALL):
                # Find all rules in the grammar

                if self.start_rule is None:
                    # Set the first rule found as the start rule.
                    self.start_rule = {"symbol": rule.group('rulename'),
                                       "type": "NT"}

                # Create and add a new rule.
                self.non_terminals[rule.group('rulename')] = {
                    'id': rule.group('rulename'),
                    'min_steps': maxsize,
                    'expanded': False,
                    'recursive': True,
                    'b_factor': 0}

                # Initialise empty list of all production choices for this
                # rule.
                tmp_productions = []

                for p in finditer(self.productionregex,
                                  rule.group('production'), MULTILINE):
                    # Iterate over all production choices for this rule.
                    # Split production choices of a rule.

                    if p.group('production') is None or p.group(
                            'production').isspace():
                        # Skip to the next iteration of the loop if the
                        # current "p" production is None or blank space.
                        continue

                    # Initialise empty data structures for production choice
                    tmp_production, terminalparts = [], None

                    # special cases: GE_RANGE and GE_GENERATE
                    GE_GENERATE_regex = r'GE_(RANGE|GENERATE):(.*)'
                    m = match(GE_GENERATE_regex, p.group('production'))
                    if m:
                        GENERATE_or_RANGE, expr = m.groups()
                        for sym in self.GE_GENERATE(GENERATE_or_RANGE, expr):

                            # print(sym)

                            # add a terminal symbol
                            tmp_production, terminalparts = [], None
                            symbol = {
                                "symbol": sym,
                                "type": "T",
                                "min_steps": 0,
                                "recursive": False}
                            tmp_production.append(symbol)
                            if sym not in self.terminals:
                                self.terminals[sym] = \
                                    [rule.group('rulename')]
                            elif rule.group('rulename') not in \
                                    self.terminals[sym]:
                                self.terminals[sym].append(
                                    rule.group('rulename'))
                            tmp_productions.append({"choice": tmp_production,
                                                    "recursive": False,
                                                    "NT_kids": False})
                        # don't try to process this production further
                        # (but later productions in same rule will work)
                        continue

                    for sub_p in finditer(self.productionpartsregex,
                                          p.group('production').strip()):
                        # Split production into terminal and non terminal
                        # symbols.

                        if sub_p.group('subrule'):
                            if terminalparts is not None:
                                # Terminal symbol is to be appended to the
                                # terminals dictionary.
                                symbol = {"symbol": terminalparts,
                                          "type": "T",
                                          "min_steps": 0,
                                          "recursive": False}
                                tmp_production.append(symbol)
                                if terminalparts not in self.terminals:
                                    self.terminals[terminalparts] = \
                                        [rule.group('rulename')]
                                elif rule.group('rulename') not in \
                                        self.terminals[terminalparts]:
                                    self.terminals[terminalparts].append(
                                        rule.group('rulename'))
                                terminalparts = None

                            tmp_production.append(
                                {"symbol": sub_p.group('subrule'),
                                 "type": "NT"})

                        else:
                            # Unescape special characters (\n, \t etc.)
                            if terminalparts is None:
                                terminalparts = ''
                            terminalparts += ''.join(
                                [part.encode().decode('unicode-escape') for
                                 part in sub_p.groups() if part])

                    if terminalparts is not None:
                        # Terminal symbol is to be appended to the terminals
                        # dictionary.
                        symbol = {"symbol": terminalparts,
                                  "type": "T",
                                  "min_steps": 0,
                                  "recursive": False}
                        tmp_production.append(symbol)
                        if terminalparts not in self.terminals:
                            self.terminals[terminalparts] = \
                                [rule.group('rulename')]
                        elif rule.group('rulename') not in \
                                self.terminals[terminalparts]:
                            self.terminals[terminalparts].append(
                                rule.group('rulename'))
                    tmp_productions.append({"choice": tmp_production,
                                            "recursive": False,
                                            "NT_kids": False})

                if not rule.group('rulename') in self.rules:
                    # Add new production rule to the rules dictionary if not
                    # already there.
                    self.rules[rule.group('rulename')] = {
                        "choices": tmp_productions,
                        "no_choices": len(tmp_productions)}

                    if len(tmp_productions) == 1:
                        # Unit productions.
                        print("Warning: Grammar contains unit production "
                              "for production rule", rule.group('rulename'))
                        print("         Unit productions consume GE codons.")
                else:
                    # Conflicting rules with the same name.
                    raise ValueError("lhs should be unique",
                                     rule.group('rulename'))

    def check_depths(self):
        """
        Run through a grammar and find out the minimum distance from each
        NT to the nearest T. Useful for initialisation methods where we
        need to know how far away we are from fully expanding a tree
        relative to where we are in the tree and what the depth limit is.

        :return: Nothing.
        """

        # Initialise graph and counter for checking minimum steps to Ts for
        # each NT.
        counter, graph = 1, []

        for rule in sorted(self.rules.keys()):
            # Iterate over all NTs.
            choices = self.rules[rule]['choices']

            # Set branching factor for each NT.
            self.non_terminals[rule]['b_factor'] = self.rules[rule][
                'no_choices']

            for choice in choices:
                # Add a new edge to our graph list.
                graph.append([rule, choice['choice']])

        while graph:
            removeset = set()
            for edge in graph:
                # Find edges which either connect to terminals or nodes
                # which are fully expanded.
                if all([sy["type"] == "T" or
                        self.non_terminals[sy["symbol"]]['expanded']
                        for sy in edge[1]]):
                    removeset.add(edge[0])

            for s in removeset:
                # These NTs are now expanded and have their correct minimum
                # path set.
                self.non_terminals[s]['expanded'] = True
                self.non_terminals[s]['min_steps'] = counter

            # Create new graph list and increment counter.
            graph = [e for e in graph if e[0] not in removeset]
            counter += 1

    def check_recursion(self, cur_symbol, seen):
        """
        Traverses the grammar recursively and sets the properties of each rule.

        :param cur_symbol: symbol to check.
        :param seen: Contains already checked symbols in the current traversal.
        :return: Boolean stating whether or not cur_symbol is recursive.
        """

        if cur_symbol not in self.non_terminals.keys():
            # Current symbol is a T.
            return False

        if cur_symbol in seen:
            # Current symbol has already been seen, is recursive.
            return True

        # Append current symbol to seen list.
        seen.append(cur_symbol)

        # Get choices of current symbol.
        choices = self.rules[cur_symbol]['choices']

        recursive = False
        for choice in choices:
            for sym in choice['choice']:
                # T is always non-recursive so no need to care about them
                if sym["type"] == "NT":
                    # Check the cache, no need to traverse the same subtree multiple times
                    if sym["symbol"] in self.recursion_cache:
                        # Grab previously calculated value
                        recursion_result = self.recursion_cache[sym["symbol"]]
                    else:
                        # Traverse subtree
                        recursion_result = self.check_recursion(sym["symbol"], seen)
                        # Add result to cache for future runs
                        self.recursion_cache[sym["symbol"]] = recursion_result

                    recursive = recursive or recursion_result

        # Set recursive properties.
        self.non_terminals[cur_symbol]['recursive'] = recursive
        seen.remove(cur_symbol)

        return recursive

    def set_arity(self):
        """
        Set the minimum path of the grammar, i.e. the smallest legal
        solution that can be generated.

        Set the maximum arity of the grammar, i.e. the longest path to a
        terminal from any non-terminal.

        :return: Nothing
        """

        # Set the minimum path of the grammar as the minimum steps to a
        # terminal from the start rule.
        self.min_path = self.non_terminals[self.start_rule["symbol"]][
            'min_steps']

        # Set the maximum arity of the grammar as the longest path to
        # a T from any NT.
        self.max_arity = max(self.non_terminals[NT]['min_steps']
                             for NT in self.non_terminals)

        # Add the minimum terminal path to each production rule.
        for rule in self.rules:
            for choice in self.rules[rule]['choices']:
                NT_kids = [i for i in choice['choice'] if i["type"] == "NT"]
                if NT_kids:
                    choice['NT_kids'] = True
                    for sym in NT_kids:
                        sym['min_steps'] = self.non_terminals[sym["symbol"]][
                            'min_steps']

        # Add boolean flag indicating recursion to each production rule.
        for rule in self.rules:
            for prod in self.rules[rule]['choices']:
                for sym in [i for i in prod['choice'] if i["type"] == "NT"]:
                    sym['recursive'] = self.non_terminals[sym["symbol"]][
                        'recursive']
                    if sym['recursive']:
                        prod['recursive'] = True

    def set_grammar_properties(self):
        """
        Goes through all non-terminals and finds the production choices with
        the minimum steps to terminals and with recursive steps.

        :return: Nothing
        """

        for nt in self.non_terminals:
            # Loop over all non terminals.
            # Find the production choices for the current NT.
            choices = self.rules[nt]['choices']

            for choice in choices:
                # Set the maximum path to a terminal for each produciton choice
                choice['max_path'] = max([item["min_steps"] for item in
                                          choice['choice']])

            # Find shortest path to a terminal for all production choices for
            # the current NT. The shortest path will be the minimum of the
            # maximum paths to a T for each choice over all choices.
            min_path = min([choice['max_path'] for choice in choices])

            # Set the minimum path in the self.non_terminals dict.
            self.non_terminals[nt]['min_path'] = [choice for choice in
                                                  choices if choice[
                                                      'max_path'] == min_path]

            # Find recursive production choices for current NT. If any
            # constituent part of a production choice is recursive,
            # it is added to the recursive list.
            self.non_terminals[nt]['recursive'] = [choice for choice in
                                                   choices if choice[
                                                       'recursive']]

    def check_permutations(self):
        """
        Calculates how many possible derivation tree combinations can be
        created from the given grammar at a specified depth. Only returns
        possible combinations at the specific given depth (if there are no
        possible permutations for a given depth, will return 0).

        :param ramps:
        :return: Nothing.
        """

        # Set the number of depths permutations are calculated for
        # (starting from the minimum path of the grammar)
        ramps = params['PERMUTATION_RAMPS']

        perms_list = []
        if self.max_arity > self.min_path:
            for i in range(max((self.max_arity + 1 - self.min_path), ramps)):
                x = self.check_all_permutations(i + self.min_path)
                perms_list.append(x)
                if i > 0:
                    perms_list[i] -= sum(perms_list[:i])
                    self.permutations[i + self.min_path] -= sum(perms_list[:i])
        else:
            for i in range(ramps):
                x = self.check_all_permutations(i + self.min_path)
                perms_list.append(x)
                if i > 0:
                    perms_list[i] -= sum(perms_list[:i])
                    self.permutations[i + self.min_path] -= sum(perms_list[:i])

    def check_all_permutations(self, depth):
        """
        Calculates how many possible derivation tree combinations can be
        created from the given grammar at a specified depth. Returns all
        possible combinations at the specific given depth including those
        depths below the given depth.

        :param depth: A depth for which to calculate the number of
        permutations of solution that can be generated by the grammar.
        :return: The permutations possible at the given depth.
        """

        if depth < self.min_path:
            # There is a bug somewhere that is looking for a tree smaller than
            # any we can create
            s = "representation.grammar.Grammar.check_all_permutations\n" \
                "Error: cannot check permutations for tree smaller than the " \
                "minimum size."
            raise Exception(s)

        if depth in self.permutations.keys():
            # We have already calculated the permutations at the requested
            # depth.
            return self.permutations[depth]

        else:
            # Calculate permutations at the requested depth.
            # Initialise empty data arrays.
            pos, depth_per_symbol_trees, productions = 0, {}, []

            for NT in self.non_terminals:
                # Iterate over all non-terminals to fill out list of
                # productions which contain non-terminal choices.
                a = self.non_terminals[NT]

                for rule in self.rules[a['id']]['choices']:
                    if rule['NT_kids']:
                        productions.append(rule)

            # Get list of all production choices from the start symbol.
            start_symbols = self.rules[self.start_rule["symbol"]]['choices']

            for choice in productions:
                # Generate a list of the symbols of each production choice
                key = str([sym['symbol'] for sym in choice['choice']])

                # Initialise permutations dictionary with the list
                depth_per_symbol_trees[key] = {}

            for i in range(2, depth + 1):
                # Find all the possible permutations from depth of min_path up
                # to a specified depth

                for choice in productions:
                    # Iterate over all production choices
                    sym_pos = 1

                    for j in choice['choice']:
                        # Iterate over all symbols in a production choice.
                        symbol_arity_pos = 0

                        if j["type"] == "NT":
                            # We are only interested in non-terminal symbols
                            for child in self.rules[j["symbol"]]['choices']:
                                # Iterate over all production choices for
                                # each NT symbol in the original choice.

                                if len(child['choice']) == 1 and \
                                        child['choice'][0]["type"] == "T":
                                    # If the child choice leads directly to
                                    # a single terminal, increment the
                                    # permutation count.
                                    symbol_arity_pos += 1

                                else:
                                    # The child choice does not lead
                                    # directly to a single terminal.
                                    # Generate a key for the permutations
                                    # dictionary and increment the
                                    # permutations count there.
                                    key = [sym['symbol'] for sym in
                                           child['choice']]
                                    if (i - 1) in depth_per_symbol_trees[
                                        str(key)].keys():
                                        symbol_arity_pos += \
                                            depth_per_symbol_trees[str(key)][
                                                i - 1]

                            # Multiply original count by new count.
                            sym_pos *= symbol_arity_pos

                    # Generate new key for the current production choice and
                    # set the new value in the permutations dictionary.
                    key = [sym['symbol'] for sym in choice['choice']]
                    depth_per_symbol_trees[str(key)][i] = sym_pos

            # Calculate permutations for the start symbol.
            for sy in start_symbols:
                key = [sym['symbol'] for sym in sy['choice']]
                if str(key) in depth_per_symbol_trees:
                    pos += depth_per_symbol_trees[str(key)][depth] if depth in \
                                                                      depth_per_symbol_trees[
                                                                          str(
                                                                              key)] else 0
                else:
                    pos += 1

            # Set the overall permutations dictionary for the current depth.
            self.permutations[depth] = pos

            return pos

    def get_min_ramp_depth(self):
        """
        Find the minimum depth at which ramping can start where we can have
        unique solutions (no duplicates).

        :param self: An instance of the representation.grammar.grammar class.
        :return: The minimum depth at which unique solutions can be generated
        """

        max_tree_depth = params['MAX_INIT_TREE_DEPTH']
        size = params['POPULATION_SIZE']

        # Specify the range of ramping depths
        depths = range(self.min_path, max_tree_depth + 1)

        if size % 2:
            # Population size is odd
            size += 1

        if size / 2 < len(depths):
            # The population size is too small to fully cover all ramping
            # depths. Only ramp to the number of depths we can reach.
            depths = depths[:int(size / 2)]

        # Find the minimum number of unique solutions required to generate
        # sufficient individuals at each depth.
        unique_start = int(floor(size / len(depths)))
        ramp = None

        for i in sorted(self.permutations.keys()):
            # Examine the number of permutations and combinations of unique
            # solutions capable of being generated by a grammar across each
            # depth i.
            if self.permutations[i] > unique_start:
                # If the number of permutations possible at a given depth i is
                # greater than the required number of unique solutions,
                # set the minimum ramp depth and break out of the loop.
                ramp = i
                break
        self.min_ramp = ramp

    def find_concatenation_NTs(self):
        """
        Scour the grammar class to find non-terminals which can be used to
        combine/reduce_trees derivation trees. Build up a list of such
        non-terminals. A concatenation non-terminal is one in which at least
        one production choice contains multiple non-terminals. For example:

            <e> ::= (<e><o><e>)|<v>

        is a concatenation NT, since the production choice (<e><o><e>) can
        reduce_trees multiple NTs together. Note that this choice also includes
        a combination of terminals and non-terminals.

        :return: Nothing.
        """

        # Iterate over all non-terminals/production rules.
        for rule in sorted(self.rules.keys()):

            # Find rules which have production choices leading to NTs.
            concat = [choice for choice in self.rules[rule]['choices'] if
                      choice['NT_kids']]

            if concat:
                # We can reduce_trees NTs.
                for choice in concat:

                    symbols = [[sym['symbol'], sym['type']] for sym in
                               choice['choice']]

                    NTs = [sym['symbol'] for sym in choice['choice'] if
                           sym['type'] == "NT"]

                    for NT in NTs:
                        # We add to our self.concat_NTs dictionary. The key is
                        # the root node we want to reduce_trees with another
                        # node. This way when we have a node and wish to see
                        # if we can reduce_trees it with anything else, we
                        # simply look up this dictionary.
                        conc = [choice['choice'], rule, symbols]

                        if NT not in self.concat_NTs:
                            self.concat_NTs[NT] = [conc]
                        else:
                            if conc not in self.concat_NTs[NT]:
                                self.concat_NTs[NT].append(conc)

    def __str__(self):
        return "%s %s %s %s" % (self.terminals, self.non_terminals,
                                self.rules, self.start_rule)

    def GE_GENERATE(self, GENERATE_or_RANGE, expr):
        """

        generate and yield strings

        GE_RANGE:dataset_n_vars  -> 0 | 1 | ... | n_vars-1
        GE_RANGE:dataset_n_is    -> 0 | 1 | ... | n_is-1
        GE_RANGE:dataset_n_os    -> 0 | 1 | ... | n_os-1
        GE_RANGE:<int>           -> 0 | 1 | ... | n-1
        
        GE_GENERATE:dataset_vars_const -> x[0] | x[1] | ... x[n-1]
        GE_GENERATE:dataset_lte_const  -> expressions of form: x[i] <= v (numerical vars)
        
        GE_GENERATE:dataset_gt   -> expressions of form: x[i] > v (numerical vars)
        GE_GENERATE:dataset_eq   -> expressions of form: x[i] == v (categorical vars)
        GE_GENERATE:dataset_neq  -> expressions of form: x[i] != v (categorical vars)
        GE_GENERATE:dataset_in   -> expressions of form: x[i].isin(s) (categorical vars; s a set)
        GE_GENERATE:dataset_nin  -> expressions of form: ~x[i].isin(s) (categorical vars; s a set)                

        GE_GENERATE:<sthg else>  -> the <sthg else> must be eval-able to give a list of strings
        
        """

        # print(f"expr: '{expr}'")
        
        if GENERATE_or_RANGE == "RANGE":
                
            try:
                if expr == "dataset_n_vars":
                    # number of columns from dataset
                    n = params['FITNESS_FUNCTION'].n_vars
                elif expr == "dataset_n_is":
                    # number of input symbols (see if_else_classifier.py)
                    n = params['FITNESS_FUNCTION'].n_is
                elif expr == "dataset_n_os":
                    # number of output symbols
                    n = params['FITNESS_FUNCTION'].n_os
                else:
                    # assume it's just an int
                    n = int(expr)
            except (ValueError, AttributeError):
                raise ValueError("Bad use of GE_RANGE: " + expr)

            for i in range(n):
                yield str(i)
                    
        else: # GENERATE_or_RANGE == "GENERATE"

            try:

                # ensure x is a Pandas DataFrame
                x = params['FITNESS_FUNCTION'].training_in 
                if type(x) != type(pd.DataFrame()):
                    x = pd.DataFrame(x)
                    
                # numerical comparisons, so dtypes[idx] != object
                if expr == "dataset_lte_const":
                    exprs = [f"x['{idx}'] <= {val}" for idx in x.columns
                             for val in sorted(x[idx].unique())
                             if x.dtypes[idx] != object]
                elif expr == "dataset_gt_const":
                    exprs = [f"x['{idx}'] > {val}" for idx in x.columns
                             for val in sorted(x[idx].unique())
                             if x.dtypes[idx] != object]

                # string comparisons, so dtypes[idx] == object
                elif expr == "dataset_eq":
                    exprs = [f"x['{idx}'] == '{val}'" for idx in x.columns
                             for val in x[idx].unique()
                             if x.dtypes[idx] == object]
                elif expr == "dataset_neq":
                    exprs = [f"x['{idx}'] != '{val}'" for idx in x.columns
                             for val in x[idx].unique()
                             if x.dtypes[idx] == object]
                elif expr == "dataset_in":
                    exprs = [f"x['{idx}'].isin({s})" for idx in x.columns
                             # no point in sets of len 0 or 1
                             for s in powerset(x[idx].unique(), minlen=2) 
                             if x.dtypes[idx] == object]
                elif expr == "dataset_nin":
                    exprs = [f"~x['{idx}'].isin({s})" for idx in x.columns
                             # no point in sets of len 0 or 1
                             for s in powerset(x[idx].unique(), minlen=2) 
                             if x.dtypes[idx] == object]

                # user code
                else:
                    exprs = eval(expr)

                assert type(exprs) == list
                assert type(exprs[0]) == str
            except:
                raise ValueError("Bad use of GE_GENERATE: " + expr +
                                 " should eval to list of str")
                
            for s in exprs: yield(s)
                    
