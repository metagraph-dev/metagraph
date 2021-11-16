from collections import defaultdict
from os import stat
import secrets
from dask import delayed
from dask import dataframe as dd
import distributed
from distributed.diagnostics.plugin import SchedulerPlugin
import pandas as pd
from dataclasses import dataclass
import numpy as np
from typing import List, Tuple
from multiprocessing import shared_memory


class CSRLoader:
    @staticmethod
    def register_dask_scheduler_plugin(client: distributed.client):
        raise NotImplementedError

    @staticmethod
    def allocate(shape, nvalues, pointers_dtype, indices_dtype, values_dtype):
        raise NotImplementedError

    @staticmethod
    def dask_incref(csr):
        raise NotImplementedError

    @staticmethod
    def load_pointers_chunk(csr, offset: int, chunk: np.ndarray):
        raise NotImplementedError

    @staticmethod
    def load_indices_chunk(csr, offset: int, chunk: np.ndarray):
        raise NotImplementedError

    @staticmethod
    def load_values_chunk(csr, offset: int, chunk: np.ndarray):
        raise NotImplementedError


### LocalCSRMatrix is for testing only


class LocalCSRMatrix:
    def __init__(self, shape, nvalues, pointers_dtype, indices_dtype, values_dtype):
        self.shape = shape
        self.pointers = np.zeros(shape=(shape[0] + 1), dtype=pointers_dtype)
        self.indices = np.zeros(shape=(nvalues,), dtype=indices_dtype)
        self.values = np.zeros(shape=(nvalues,), dtype=values_dtype)

    def __str__(self):
        return f"shape: {self.shape}\npointers{self.pointers}\nindices: {self.indices}\nvalues: {self.values}"


class LocalCSRLoader:
    @staticmethod
    def register_dask_scheduler_plugin(client: distributed.Client):
        # nothing to do since Python refcount works fine for LocalCSRMatrix
        pass

    @staticmethod
    def allocate(shape, nvalues, pointers_dtype, indices_dtype, values_dtype):
        return LocalCSRMatrix(
            shape, nvalues, pointers_dtype, indices_dtype, values_dtype
        )

    @staticmethod
    def dask_incref(csr):
        # nothing to do since Python refcount works fine for LocalCSRMatrix
        pass

    @staticmethod
    def load_pointers_chunk(csr, offset: int, chunk: np.ndarray):
        csr.pointers[offset : offset + len(chunk)] = chunk

    @staticmethod
    def load_indices_chunk(csr, offset: int, chunk: np.ndarray):
        csr.indices[offset : offset + len(chunk)] = chunk

    @staticmethod
    def load_values_chunk(csr, offset: int, chunk: np.ndarray):
        csr.values[offset : offset + len(chunk)] = chunk


### SharedCSRMatrix is for testing only


class SharedCSRMatrix:
    TAG_POINTERS = "pointers"
    TAG_INDICES = "indices"
    TAG_VALUES = "values"

    def __init__(self, shape, nvalues, pointers_dtype, indices_dtype, values_dtype):
        self.shape = shape
        self.shm_name = "mg-" + secrets.token_hex(8)
        self.pointers_shm, self.pointers = self._get_shared_array(
            self.TAG_POINTERS, shape=(shape[0] + 1,), dtype=pointers_dtype, create=True
        )
        self.indices_shm, self.indices = self._get_shared_array(
            self.TAG_INDICES, shape=(nvalues,), dtype=indices_dtype, create=True
        )
        self.values_shm, self.values = self._get_shared_array(
            self.TAG_VALUES, shape=(nvalues,), dtype=values_dtype, create=True
        )

    def _get_shared_array(
        self, tag: str, shape: Tuple[int], dtype: np.dtype, create: bool
    ):
        n = shape[0]
        shm = shared_memory.SharedMemory(
            name=self.shm_name + tag, create=create, size=n * np.dtype(dtype).itemsize
        )
        ary = np.ndarray(n, dtype=dtype, buffer=shm.buf)
        return shm, ary

    def __getstate__(self):
        return {
            "shape": self.shape,
            "shm_name": self.shm_name,
            "pointers": {
                "tag": self.TAG_POINTERS,
                "shape": self.pointers.shape,
                "dtype": self.pointers.dtype,
            },
            "indices": {
                "tag": self.TAG_INDICES,
                "shape": self.indices.shape,
                "dtype": self.indices.dtype,
            },
            "values": {
                "tag": self.TAG_VALUES,
                "shape": self.values.shape,
                "dtype": self.values.dtype,
            },
        }

    def _setstate_connect_shared_ndarray(self, array_desc):
        tag = array_desc["tag"]
        shape = array_desc["shape"]
        n = shape[0]
        dtype = array_desc["dtype"]

        return self._get_shared_array(tag, shape, dtype, create=False)

    def __setstate__(self, state):
        self.shape = state["shape"]
        self.shm_name = state["shm_name"]
        self.pointers_shm, self.pointers = self._setstate_connect_shared_ndarray(
            state["pointers"]
        )
        self.indices_shm, self.indices = self._setstate_connect_shared_ndarray(
            state["indices"]
        )
        self.values_shm, self.values = self._setstate_connect_shared_ndarray(
            state["values"]
        )

    def __str__(self):
        return f"shape: {self.shape}\npointers{self.pointers}\nindices: {self.indices}\nvalues: {self.values}"


class SharedMemoryRefCounter(SchedulerPlugin):
    def __init__(self, tag):
        self.tag = tag
        self.shmem_to_key = defaultdict(lambda: set())
        self.key_to_shmem = defaultdict(lambda: set())

    def transition(self, key, start, finish, *args, **kwargs):
        if start == "released" and key.startswith(self.tag):
            _, msg_key, msg_shmem = key.split(":")
            self.key_to_shmem[msg_key].add(msg_shmem)
            self.shmem_to_key[msg_shmem].add(msg_key)

        if finish == "forgotten":
            if key in self.key_to_shmem:
                for shmem in self.key_to_shmem[key]:
                    self.shmem_to_key[shmem].remove(key)
                    if len(self.shmem_to_key[shmem]) == 0:
                        sh = shared_memory.SharedMemory(shmem, create=False)
                        sh.unlink()
                        del self.shmem_to_key[shmem]
                del self.key_to_shmem[key]


class SharedCSRLoader:
    REFCOUNT_TAG = "shared_csr_loader_refcount"

    @classmethod
    def register_dask_scheduler_plugin(cls, client: distributed.Client):
        plugin = SharedMemoryRefCounter(cls.REFCOUNT_TAG)
        client.register_scheduler_plugin(
            plugin, idempotent=True, name="metagraph_shared_csr_refcount"
        )

    @staticmethod
    def allocate(shape, nvalues, pointers_dtype, indices_dtype, values_dtype):
        csr = SharedCSRMatrix(
            shape, nvalues, pointers_dtype, indices_dtype, values_dtype
        )
        return csr

    @classmethod
    def dask_incref(cls, csr):
        def shared_csr_loader_incref(x):
            # This does nothing.  Exists only to trick scheduler into generating an event
            pass

        key = distributed.get_worker().get_current_task()
        client = distributed.get_client()

        for shm in [csr.pointers_shm, csr.indices_shm, csr.values_shm]:
            task_name = f"{cls.REFCOUNT_TAG}:{key}:{shm.name}"
            dummy_arg = key + shm.name
            client.submit(
                shared_csr_loader_incref, dummy_arg, key=task_name, pure=False
            )

    @staticmethod
    def load_pointers_chunk(csr, offset: int, chunk: np.ndarray):
        csr.pointers[offset : offset + len(chunk)] = chunk

    @staticmethod
    def load_indices_chunk(csr, offset: int, chunk: np.ndarray):
        csr.indices[offset : offset + len(chunk)] = chunk

    @staticmethod
    def load_values_chunk(csr, offset: int, chunk: np.ndarray):
        csr.values[offset : offset + len(chunk)] = chunk


### Generic COO to CSR Loading logic


@dataclass
class COODescriptor:
    matrix_shape: Tuple[int, int]
    row_fieldname: str
    col_fieldname: str
    value_fieldname: str


@dataclass
class COOChunkInfo:
    # Assumes row-sorted chunks
    partition_id: int
    matrix_shape: Tuple[int, int]
    nvalues: int
    first_row: int
    last_row: int
    row_dtype: np.dtype
    col_dtype: np.dtype
    value_dtype: np.dtype


@dataclass
class COOtoCSRChunkPlan:
    coo_chunk: COOChunkInfo
    fill_row_begin: int
    fill_row_end: int
    index_value_offset: int  # offset into CSR index and value arrays


@dataclass
class COOtoCSRPlan:
    coo_desc: COODescriptor
    matrix_shape: Tuple[int, int]
    nvalues: int
    pointer_dtype: np.dtype
    index_dtype: np.dtype
    value_dtype: np.dtype
    chunks: List[COOtoCSRChunkPlan]


@delayed
def extract_chunk_information(
    partition_id: int, partition: pd.DataFrame, coo_desc: COODescriptor
) -> COOChunkInfo:
    matrix_rows, matrix_cols = coo_desc.matrix_shape
    nvalues = len(partition)
    first_row = partition[coo_desc.row_fieldname].iloc[0]
    last_row = partition[coo_desc.row_fieldname].iloc[-1]

    return COOChunkInfo(
        partition_id=partition_id,
        matrix_shape=(matrix_rows, matrix_cols),
        nvalues=nvalues,
        first_row=first_row,
        last_row=last_row,
        row_dtype=partition.dtypes[coo_desc.row_fieldname],
        col_dtype=partition.dtypes[coo_desc.col_fieldname],
        value_dtype=partition.dtypes[coo_desc.value_fieldname],
    )


@delayed
def build_plan(coo_desc: COODescriptor, chunks: List[COOChunkInfo]) -> COOtoCSRPlan:
    if len(chunks) == 0:
        raise ValueError("must be at least one chunk in input COO data")

    chunks = sorted(chunks, key=lambda x: x.first_row)

    # Use first chunk to seed attributes.  Will verify consistency with remaining chunks
    first_chunk = chunks[0]
    index_dtype = np.result_type(first_chunk.row_dtype, first_chunk.col_dtype)
    value_dtype = first_chunk.value_dtype
    plan = COOtoCSRPlan(
        coo_desc=coo_desc,
        matrix_shape=first_chunk.matrix_shape,
        pointer_dtype=index_dtype,
        index_dtype=index_dtype,
        value_dtype=value_dtype,
        nvalues=0,  # recompute below
        chunks=[],
    )

    # scan chunks to compute totals and offsets
    nvalues = 0
    last_row = -1
    prev_chunk_plan = None
    for chunk in chunks:
        # consistency check
        if chunk.matrix_shape != plan.matrix_shape:
            raise ValueError(
                f"chunk {chunk.filename} has matrix shape {chunk.matrix_shape} inconsistent with current shape {plan.matrix_shape}"
            )
        if chunk.first_row <= last_row:
            raise ValueError(
                f"chunk {chunk.filename} has row overlap with another chunk"
            )

        plan.pointer_dtype = np.result_type(
            plan.pointer_dtype, chunk.row_dtype, chunk.col_dtype
        )
        plan.index_dtype = np.result_type(
            plan.index_dtype, chunk.row_dtype, chunk.col_dtype
        )
        plan.value_dtype = np.result_type(plan.value_dtype, chunk.value_dtype)

        chunk_plan = COOtoCSRChunkPlan(
            coo_chunk=chunk,
            fill_row_begin=chunk.first_row,
            fill_row_end=plan.matrix_shape[0] - 1,  # will be replaced on next iteration
            index_value_offset=nvalues,
        )
        if prev_chunk_plan is not None:
            prev_chunk_plan.fill_row_end = chunk.first_row - 1
        prev_chunk_plan = chunk_plan

        plan.chunks.append(chunk_plan)
        nvalues += chunk.nvalues

    plan.nvalues = nvalues

    return plan


@delayed(pure=False)
def allocate_csr(csr_loader: CSRLoader, plan: COOtoCSRPlan):
    csr = csr_loader.allocate(
        plan.matrix_shape,
        plan.nvalues,
        plan.pointer_dtype,
        plan.index_dtype,
        plan.value_dtype,
    )

    csr_loader.dask_incref(csr)
    return csr


@delayed(pure=False)
def load_chunk(
    csr_loader: CSRLoader,
    partition_id: int,
    partition: pd.DataFrame,
    plan: COOtoCSRPlan,
    csr,
) -> str:
    coo_desc = plan.coo_desc
    chunk_plan = plan.chunks[partition_id]

    # we assume the records were already sorted by row number, but this ensures the column numbers are sorted within the row
    partition.sort_values(by=[coo_desc.row_fieldname, coo_desc.col_fieldname])

    # compute pointer offsets
    rows = partition[coo_desc.row_fieldname].to_numpy()
    unique_rows, elements_per_row = np.unique(rows, return_counts=True)
    pointers = np.zeros(
        shape=(chunk_plan.fill_row_end - chunk_plan.fill_row_begin + 1),
        dtype=plan.pointer_dtype,
    )
    pointers[unique_rows - chunk_plan.fill_row_begin] = elements_per_row
    pointers = np.cumsum(pointers) + chunk_plan.index_value_offset

    # copy indices and values
    csr_loader.load_pointers_chunk(csr, chunk_plan.fill_row_begin + 1, pointers)
    csr_loader.load_indices_chunk(
        csr, chunk_plan.index_value_offset, partition[coo_desc.col_fieldname].to_numpy()
    )
    csr_loader.load_values_chunk(
        csr,
        chunk_plan.index_value_offset,
        partition[coo_desc.value_fieldname].to_numpy(),
    )

    # return dummy value
    return partition_id


@delayed(pure=False)
def finalize_csr(csr_loader: CSRLoader, csr, load_chunk_results: list):
    csr_loader.dask_incref(csr)
    return csr


def load_coo_to_csr(
    coo: dd.DataFrame,
    shape: Tuple[int, int],
    loader: CSRLoader,
    row="row",
    col="col",
    value="value",
    client=None,
):
    if client is None:
        client = distributed.get_client()
    loader.register_dask_scheduler_plugin(client)

    coo_desc = COODescriptor(shape, row, col, value)
    chunks = [
        extract_chunk_information(i, part, coo_desc)
        for i, part in enumerate(coo.partitions)
    ]
    plan = build_plan(coo_desc, chunks)
    empty_csr = allocate_csr(loader, plan)
    loaded_chunks = [
        load_chunk(loader, i, part, plan, empty_csr)
        for i, part in enumerate(coo.partitions)
    ]
    csr = finalize_csr(loader, empty_csr, loaded_chunks)

    return csr
