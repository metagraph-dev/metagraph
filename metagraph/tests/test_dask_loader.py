from metagraph.core.dask import loader
import pandas as pd
import numpy as np
import dask.dataframe as dd
import distributed
import dataclasses
import pytest
import pickle


def test_loader_base_exceptions():
    load_base = loader.CSRLoader()

    with pytest.raises(NotImplementedError):
        load_base.register_dask_scheduler_plugin(None)
    with pytest.raises(NotImplementedError):
        load_base.allocate((2, 2), 100, np.int64, np.int64, np.float64)
    with pytest.raises(NotImplementedError):
        load_base.dask_incref(None)
    with pytest.raises(NotImplementedError):
        load_base.load_chunk(None, 10, np.arange(10), 20, np.arange(10), np.arange(10))
    with pytest.raises(NotImplementedError):
        load_base.finalize(None, [])


def test_extract_chunk_information(ex_coo_desc, ex_ddf):
    result0 = loader.extract_chunk_information(
        0, ex_ddf.partitions[0], ex_coo_desc
    ).compute()
    expected0 = loader.COOChunkInfo(
        partition_id=0,
        matrix_shape=(10, 10),
        nvalues=2,
        first_row=0,
        last_row=0,
        row_dtype=np.int64,
        col_dtype=np.int64,
        value_dtype=np.float64,
    )
    assert result0 == expected0

    result1 = loader.extract_chunk_information(
        1, ex_ddf.partitions[1], ex_coo_desc
    ).compute()
    expected1 = dataclasses.replace(
        expected0, partition_id=1, nvalues=3, first_row=4, last_row=4,
    )
    assert result1 == expected1

    result2 = loader.extract_chunk_information(
        2, ex_ddf.partitions[2], ex_coo_desc
    ).compute()
    expected2 = dataclasses.replace(
        expected0, partition_id=2, nvalues=2, first_row=8, last_row=9,
    )
    assert result2 == expected2


def test_build_plan(ex_coo_desc, ex_chunks, ex_coo_to_csr_plan):
    result = loader.build_plan(ex_coo_desc, ex_chunks).compute()
    expected = ex_coo_to_csr_plan
    assert result == expected

    return expected


def test_build_plan_empty(ex_coo_desc):
    with pytest.raises(ValueError, match="must be at least one chunk"):
        loader.build_plan(ex_coo_desc, []).compute()


def test_build_plan_inconsistent_shape(ex_coo_desc, ex_chunks):
    ex_chunks[1].matrix_shape = (20, 20)
    with pytest.raises(ValueError, match="inconsistent with current shape"):
        loader.build_plan(ex_coo_desc, ex_chunks).compute()


def test_build_plan_row_overlap(ex_coo_desc, ex_chunks):
    ex_chunks[1].first_row = 0
    with pytest.raises(ValueError, match="row overlap"):
        loader.build_plan(ex_coo_desc, ex_chunks).compute()


@pytest.mark.parametrize("csr_loader", [loader.SharedCSRLoader])
def test_allocate_csr(dask_client, csr_loader, ex_coo_to_csr_plan):
    csr = loader.allocate_csr(csr_loader, ex_coo_to_csr_plan).compute()
    assert csr.shape == (10, 10)


@pytest.mark.parametrize("csr_loader", [loader.SharedCSRLoader])
def test_finalize_csr(dask_client, csr_loader, ex_coo_to_csr_plan):
    csr = loader.allocate_csr(csr_loader, ex_coo_to_csr_plan).compute()
    finalize_csr = loader.finalize_csr(csr_loader, csr, [0, 1, 2]).compute()
    assert finalize_csr.shape == (10, 10)


@pytest.mark.parametrize("csr_loader", [loader.SharedCSRLoader])
def test_load_chunk(dask_client, csr_loader, ex_ddf, ex_coo_to_csr_plan):
    csr = loader.allocate_csr(csr_loader, ex_coo_to_csr_plan).compute()

    result = loader.load_chunk(
        csr_loader, 0, ex_ddf.partitions[0], ex_coo_to_csr_plan, csr
    ).compute()
    assert (
        result is not None
    )  # don't care what this returns, as long as it is something

    np.testing.assert_equal(csr.pointers, [0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0])
    np.testing.assert_equal(csr.indices, [1, 3, 0, 0, 0, 0, 0])
    np.testing.assert_equal(csr.values, [1, 2, 0, 0, 0, 0, 0])

    result = loader.load_chunk(
        csr_loader, 1, ex_ddf.partitions[1], ex_coo_to_csr_plan, csr
    ).compute()
    assert (
        result is not None
    )  # don't care what this returns, as long as it is something

    np.testing.assert_equal(csr.pointers, [0, 2, 2, 2, 2, 5, 5, 5, 5, 0, 0])
    np.testing.assert_equal(csr.indices, [1, 3, 0, 3, 5, 0, 0])
    np.testing.assert_equal(csr.values, [1, 2, 3, 4, 5, 0, 0])

    result = loader.load_chunk(
        csr_loader, 2, ex_ddf.partitions[2], ex_coo_to_csr_plan, csr
    ).compute()
    assert (
        result is not None
    )  # don't care what this returns, as long as it is something

    np.testing.assert_equal(csr.pointers, [0, 2, 2, 2, 2, 5, 5, 5, 5, 6, 7])
    np.testing.assert_equal(csr.indices, [1, 3, 0, 3, 5, 7, 7])
    np.testing.assert_equal(csr.values, [1, 2, 3, 4, 5, 6, 7])


def test_shared_csr_pickle(ex_coo_to_csr_plan):
    csr = loader.allocate_csr(loader.SharedCSRLoader, ex_coo_to_csr_plan).compute()
    loader.SharedCSRLoader.load_chunk(
        csr,
        0,
        [0, 2, 2, 2, 2, 5, 5, 5, 5, 6, 7],
        0,
        [1, 3, 0, 3, 5, 7, 7],
        [1, 2, 3, 4, 5, 6, 7],
    )

    csr_2 = pickle.loads(pickle.dumps(csr))
    np.testing.assert_equal(csr_2.pointers, [0, 2, 2, 2, 2, 5, 5, 5, 5, 6, 7])
    np.testing.assert_equal(csr_2.indices, [1, 3, 0, 3, 5, 7, 7])
    np.testing.assert_equal(csr_2.values, [1, 2, 3, 4, 5, 6, 7])


@pytest.mark.parametrize("csr_loader", [loader.SharedCSRLoader])
def test_load_coo_to_csr(dask_client, csr_loader, ex_ddf):
    csr = loader.load_coo_to_csr(
        ex_ddf, shape=(10, 10), row="row", col="col", value="value", loader=csr_loader,
    )

    csr.visualize(filename="test_load_coo_to_csr_" + csr_loader.__name__ + ".png")
    csr = csr.compute()

    np.testing.assert_equal(csr.pointers, [0, 2, 2, 2, 2, 5, 5, 5, 5, 6, 7])
    np.testing.assert_equal(csr.indices, [1, 3, 0, 3, 5, 7, 7])
    np.testing.assert_equal(csr.values, [1, 2, 3, 4, 5, 6, 7])

    assert (
        str(csr)
        == """shape: (10, 10)
pointers: [0 2 2 2 2 5 5 5 5 6 7]
indices: [1 3 0 3 5 7 7]
values: [1. 2. 3. 4. 5. 6. 7.]"""
    )


@pytest.fixture
def ex_ddf():
    df = pd.DataFrame(
        {
            "row": np.array([0, 0, 4, 4, 4, 8, 9], dtype=np.int64),
            "col": np.array([1, 3, 0, 3, 5, 7, 7], dtype=np.int64),
            "value": np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.float64),
        }
    )
    ddf = dd.from_pandas(df, npartitions=1).repartition(divisions=[0, 2, 5, 6])
    return ddf


@pytest.fixture
def ex_coo_desc():
    return loader.COODescriptor((10, 10), "row", "col", "value")


@pytest.fixture
def ex_chunks():
    chunk_template = loader.COOChunkInfo(
        partition_id=0,  # will be replaced
        matrix_shape=(10, 10),
        nvalues=0,  # will be replaced
        first_row=0,  # will be replaced
        last_row=0,  # will be replaced
        row_dtype=np.int64,
        col_dtype=np.int64,
        value_dtype=np.float64,
    )

    chunks = [
        dataclasses.replace(
            chunk_template, partition_id=0, nvalues=2, first_row=0, last_row=0,
        ),
        dataclasses.replace(
            chunk_template, partition_id=1, nvalues=3, first_row=4, last_row=4,
        ),
        dataclasses.replace(
            chunk_template, partition_id=2, nvalues=2, first_row=8, last_row=9,
        ),
    ]

    return chunks


@pytest.fixture
def ex_coo_to_csr_plan(ex_coo_desc, ex_chunks):
    return loader.COOtoCSRPlan(
        coo_desc=ex_coo_desc,
        matrix_shape=(10, 10),
        pointer_dtype=np.int64,
        index_dtype=np.int64,
        value_dtype=np.float64,
        nvalues=7,
        chunks=[
            loader.COOtoCSRChunkPlan(
                coo_chunk=ex_chunks[0],
                fill_row_begin=0,
                fill_row_end=3,
                index_value_offset=0,
            ),
            loader.COOtoCSRChunkPlan(
                coo_chunk=ex_chunks[1],
                fill_row_begin=4,
                fill_row_end=7,
                index_value_offset=2,
            ),
            loader.COOtoCSRChunkPlan(
                coo_chunk=ex_chunks[2],
                fill_row_begin=8,
                fill_row_end=9,
                index_value_offset=5,
            ),
        ],
    )


@pytest.fixture(scope="module")
def dask_client():
    cluster = distributed.LocalCluster(n_workers=3, threads_per_worker=1)
    client = distributed.Client(cluster)

    yield client

    client.close()
    cluster.close()
