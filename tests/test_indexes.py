# Copyright (C) 2026 Ujjwal Sharma and Omar Shahbaz Khan
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Tests for FaissIndex.incremental_search's skip_ids handling.

Covers the cases where the first search round comes up short. 
Results must not repeat across rounds, fewer than k available unskipped 
items must return what exists rather than widening the scope without bound,
and widening must not leak into the shared index's own nprobe/efSearch.

The indexes are built in a separate process and loaded through
FaissIndex.load_index, the same way the app gets them. 
Building in this process would crash on macOS, since torch 
(imported while pytest collects the model tests) and faiss each 
bundle their own OpenMP runtime, and faiss's index building is 
what starts its OpenMP threads.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from app.core.indexes import FaissIndex


DIM = 8
N = 5000
# Above load_index's default nprobe of 64, so IVF has room to widen
NLIST = 128

BUILD_SCRIPT = """
import sys
from pathlib import Path

import faiss
import numpy as np

out, dim, n, nlist = Path(sys.argv[1]), *map(int, sys.argv[2:])
x = np.random.default_rng(0).standard_normal((n, dim)).astype("f4")

hnsw = faiss.IndexHNSWFlat(dim, 16)
hnsw.add(x)
faiss.write_index(hnsw, str(out / "hnsw.faiss"))

ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(dim), dim, nlist)
ivf.train(x)
ivf.add(x)
faiss.write_index(ivf, str(out / "ivf.faiss"))

flat = faiss.IndexFlatL2(dim)
flat.add(x)
faiss.write_index(flat, str(out / "unknown.faiss"))
"""


@pytest.fixture(scope="module")
def index_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("faiss")
    subprocess.run(
        [sys.executable, "-c", BUILD_SCRIPT, str(out), str(DIM), str(N), str(NLIST)],
        check=True,
    )
    return out


@pytest.fixture(params=["hnsw", "ivf", "unknown"])
def index(request, index_dir) -> FaissIndex:
    index = FaissIndex()
    index.load_index(index_dir / f"{request.param}.faiss")
    assert index.index_type == request.param
    return index


@pytest.fixture
def query():
    return np.random.default_rng(1).standard_normal(DIM).astype("f4")


def _scope(index: FaissIndex):
    if index.index_type == "hnsw":
        return index.index.hnsw.efSearch
    if index.index_type == "ivf":
        return index.index.nprobe
    return None


def test_skipped_top_hits_are_replaced_without_duplicates(index, query):
    _, top, _ = index.search(query, 10)
    skip = set(top[:5])

    _, result, _ = index.incremental_search(query, 10, skip)

    assert len(result) == 10
    assert len(set(result)) == 10
    assert skip.isdisjoint(result)


def test_fewer_unskipped_than_k_returns_what_exists(index, query):
    skip = set(range(3, N))

    _, result, _ = index.incremental_search(query, 10, skip)

    assert sorted(result) == [0, 1, 2]


def test_everything_skipped_returns_nothing(index, query):
    _, result, _ = index.incremental_search(query, 5, set(range(N)))
    assert result == []


def test_widening_leaves_the_index_scope_untouched(index, query):
    before = _scope(index)

    index.incremental_search(query, 10, set(range(3, N)))

    assert _scope(index) == before
