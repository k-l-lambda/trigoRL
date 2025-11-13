"""
Utility functions for data processing.
"""

import re
from typing import List, Tuple


def parse_split(split: str) -> Tuple[List[int], int, bool]:
	"""
	Parse split string into phases, cycle, and shuffle flag.

	Supports both comma-separated phases and range syntax with '..'.

	Args:
	    split: Split specification string (e.g., "*0..2/5" or "3,4/5")

	Returns:
	    Tuple of (phases, cycle, shuffle)
	    - phases: List of phase indices to include
	    - cycle: Total number of phases
	    - shuffle: Whether to shuffle the dataset

	Examples:
	    >>> parse_split("*0..2/5")
	    ([0, 1, 2], 5, True)
	    >>> parse_split("3,4/5")
	    ([3, 4], 5, False)
	    >>> parse_split("*0..3,7,8/10")
	    ([0, 1, 2, 3, 7, 8], 10, True)
	    >>> parse_split("0..2,5..7/10")
	    ([0, 1, 2, 5, 6, 7], 10, False)
	"""
	shuffle = split.startswith('*')
	if shuffle:
		split = split[1:]

	phases_str, cycle_str = split.split('/')

	# Parse phases: support comma-separated list with optional range syntax
	parts = phases_str.split(',')
	phases = []

	for part in parts:
		# Check if this part is a range (e.g., "0..7")
		range_match = re.match(r'^(\d+)\.\.(\d+)$', part)
		if range_match:
			# Expand range: "0..7" -> [0,1,2,3,4,5,6,7]
			start = int(range_match.group(1))
			end = int(range_match.group(2))
			phases.extend(range(start, end + 1))
		else:
			# Single phase number
			phases.append(int(part))

	cycle = int(cycle_str)

	return phases, cycle, shuffle
