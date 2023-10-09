import numpy as np

def clean_path(path):
	""" utility function that performs basic text cleaning on path """

	# No need to modify
	path = str(path).replace("'","")
	path = path.replace(",","")
	path = path.replace(" ","")
	path = path.replace("[","")
	path = path.replace("]","")

	return path


class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        """

        self.symbol_set = symbol_set


    def decode(self, y_probs):
        """

        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """

        decoded_path = []
        blank = 0
        path_prob = 1

        for t in range(len(y_probs[0])):
            max = y_probs[0][t]
            max_idx = 0
            for j in range(1, len(y_probs)):
                if max < y_probs[j][t]:
                    max = y_probs[j][t]
                    max_idx = j
            path_prob *= max

            if max_idx != 0:
            	decoded_path.append(self.symbol_set[max_idx-1])
            else:
                decoded_path.append(blank)
        # print(decoded_path)
        compressed_path = []
        for i in range(0, len(decoded_path)):
            if i == 0:
                if decoded_path[i] != 0:
                    compressed_path.append(decoded_path[i])
            else:
                if decoded_path[i-1] != decoded_path[i] and decoded_path[i] != 0:
                    compressed_path.append(decoded_path[i])

        decoded_path = clean_path(compressed_path)

        return decoded_path, path_prob


class BeamSearchDecoder(object):

    def __init__(self, symbol_set, beam_width):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion

        """

        self.symbol_set = symbol_set
        self.beam_width = beam_width

    def InitializePaths(self, SymbolSet, y):

        InitialBlankPathScore, InitialPathScore = {}, {}
        InitialPathsWithFinalSymbol, InitialPathsWithFinalBlank = [], []
        # First push the blank into a path-ending-with-blank stack. No symbol has been invoked yet
        path = ""
        InitialBlankPathScore[path]=y[0] #Score of blank at t=1
        InitialPathsWithFinalBlank.append(path)

        # Push rest of the symbols into a path-ending-with-symbol stack
        for i, symbol in enumerate(SymbolSet): #Thisistheentiresymbolset,withouttheblank
            InitialPathScore[symbol] = y[i + 1] #Score of symbol c at t=1
            InitialPathsWithFinalSymbol.append(symbol) # Set addition

        return InitialPathsWithFinalBlank, InitialPathsWithFinalSymbol, InitialBlankPathScore, InitialPathScore

    def ExtendWithBlank(self, PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore, y):
        UpdatedPathsWithTerminalBlank = []
        UpdatedBlankPathScore = {}
        # First work on paths with terminal blanks
        #(This represents transitions along horizontal trellis edges for blanks)

        for path in PathsWithTerminalBlank:
            # Repeating a blank doesn’t change the symbol sequence
            UpdatedPathsWithTerminalBlank.append(path) # Set addition
            UpdatedBlankPathScore[path] = BlankPathScore[path] * y[0]

        # Then extend paths with terminal symbols by blanks
        for path in PathsWithTerminalSymbol:
            # If there is already an equivalent string in UpdatesPathsWithTerminalBlank
            # simply add the score. If not create a new entry
            if path in UpdatedPathsWithTerminalBlank:
                UpdatedBlankPathScore[path] += PathScore[path] * y[0]
            else:
                UpdatedPathsWithTerminalBlank.append(path) # Set addition
                UpdatedBlankPathScore[path] = PathScore[path] * y[0]

        return UpdatedPathsWithTerminalBlank, UpdatedBlankPathScore

    def ExtendWithSymbol(self, PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore, SymbolSet, y):
        UpdatedPathsWithTerminalSymbol = []
        UpdatedPathScore = {}
        # First extend the paths terminating in blanks. This will always create a new sequence
        for path in PathsWithTerminalBlank:
            for i, symbol in enumerate(SymbolSet):
                newpath = path + symbol #Concatenation
                UpdatedPathsWithTerminalSymbol.append(newpath) # Set addition
                UpdatedPathScore[newpath] = BlankPathScore[path] * y[i+1]

        # Next work on paths with terminal symbols
        for path in PathsWithTerminalSymbol:
            # Extend the path with every symbol other than blank
            for i, symbol in enumerate(SymbolSet):
                if symbol == path[-1]:
                    newpath = path
                else:
                    newpath = path + symbol

                if newpath in UpdatedPathsWithTerminalSymbol:
                    UpdatedPathScore[newpath] += PathScore[path] * y[i+1]
                else: # Create new path
                    UpdatedPathsWithTerminalSymbol.append(newpath) # Set addition
                    UpdatedPathScore[newpath] = PathScore[path] * y[i+1]

        return UpdatedPathsWithTerminalSymbol, UpdatedPathScore

    def Prune(self, PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore, BeamWidth):
        PrunedBlankPathScore, PrunedPathScore = {}, {}

        scorelist = []
        # First gather all the relevant scores
        for p in PathsWithTerminalBlank:
            scorelist.append(BlankPathScore[p])
        for p in PathsWithTerminalSymbol:
            scorelist.append(PathScore[p])

        # Sort and find cutoff score that retains exactly BeamWidth paths
        scorelist = sorted(scorelist, reverse = True) # In decreasing order

        if BeamWidth < len(scorelist):
            cutoff = scorelist[BeamWidth]
        else:
            cutoff = scorelist[-1]

        PrunedPathsWithTerminalBlank = []
        for p in PathsWithTerminalBlank:
            if BlankPathScore[p] > cutoff:
                PrunedPathsWithTerminalBlank.append(p) # Set addition
                PrunedBlankPathScore[p] = BlankPathScore[p]

        PrunedPathsWithTerminalSymbol = []
        for p in PathsWithTerminalSymbol:
            if PathScore[p] > cutoff:
                PrunedPathsWithTerminalSymbol.append(p) # Set addition
                PrunedPathScore[p] = PathScore[p]
        return PrunedPathsWithTerminalBlank, PrunedPathsWithTerminalSymbol, PrunedBlankPathScore, PrunedPathScore

    def MergeIdenticalPaths(self, PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore):
        # All paths with terminal symbosl will remain
        MergedPaths = PathsWithTerminalSymbol
        FinalPathScore = PathScore

        # Paths with terminal blanks will contribute scores to existing identical paths from
        # PathsWithTerminalSymbol if present, or be included in the final set, otherwise
        for p in PathsWithTerminalBlank:
            if p in MergedPaths:
                FinalPathScore[p] += BlankPathScore[p]
            else:
                MergedPaths.append(p)
                FinalPathScore[p] = BlankPathScore[p]

        return MergedPaths, FinalPathScore

    def decode(self, y_probs):
        """

        Perform beam search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
			batch size for part 1 will remain 1, but if you plan to use your
			implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        forward_path [str]:
            the symbol sequence with the best path score (forward probability)

        merged_path_scores [dict]:
            all the final merged paths with their scores

        """

        PathScore = {} # dict of scores for paths ending with symbols
        BlankPathScore = {} # dict of scores for paths ending with blanks
        num_symbols, seq_len, batch_size = y_probs.shape

        # First time instant: initialize paths with each of the symbols, including blank, using score at t=1
        NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol, NewBlankPathScore, NewPathScore = self.InitializePaths(self.symbol_set, y_probs[:, 0, :])

        # Subsequent time steps
        for t in range(1, seq_len):
            # Prune the collection down to the BeamWidth
            PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore = self.Prune(NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol,
                                                                                                NewBlankPathScore, NewPathScore, self.beam_width)


            # First extend paths by a blank
            NewPathsWithTerminalBlank, NewBlankPathScore =  self.ExtendWithBlank(PathsWithTerminalBlank, PathsWithTerminalSymbol,
                                                                            BlankPathScore, PathScore, y_probs[:, t, :])

            # Next extend paths by a symbol
            NewPathsWithTerminalSymbol, NewPathScore = self.ExtendWithSymbol(PathsWithTerminalBlank, PathsWithTerminalSymbol,
                                                                        BlankPathScore, PathScore, self.symbol_set, y_probs[:, t, :])

        # Merge identical paths differing only by the final blank
        MergedPaths, FinalPathScore = self.MergeIdenticalPaths(NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol, NewBlankPathScore, NewPathScore)


        # Pick the best path
        BestPath = max(FinalPathScore, key=lambda x: FinalPathScore[x]) # Find the path with the best score

        return BestPath, FinalPathScore
