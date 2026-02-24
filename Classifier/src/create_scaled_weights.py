import itertools as itt
import uproot
import logging
import pathlib
import pickle
import random
from collections import defaultdict
from dataclasses import KW_ONLY, dataclass, is_dataclass
from functools import partial
from typing import (Any, Callable, Dict, Iterable, List, Protocol, Tuple,
                    Union, runtime_checkable)

import numpy as np
import pandas as pd
import torch as t
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, StandardScaler, scale
from tqdm import tqdm
from contextlib import contextmanager
from typing import (Any, Callable, Dict, Generator, Iterable, Iterator, List,
                    Optional, Tuple, Type, Union, get_args, get_origin)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


