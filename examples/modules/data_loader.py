import pytest

from libuq import OutputExtractor, OutputType
from ecoli.library.parquet_emitter import create_duckdb_conn, dataset_sql


def test_data_loader():
    conn = create_duckdb_conn()
    sim_base_path = "/Users/alexanderpatrie/sms/vEcoli-private/api_integration/sims"
    history_sql, config_sql, _ = dataset_sql(sim_base_path, ["mecillinam"])

    extractor = OutputExtractor(conn, history_sql, config_sql)

    # Extract specific outputs
    transcriptome, cistron_ids = extractor.extract_transcriptome(
        generation_lower_bound=2,  # Skip initial generations
        time_lower_bound=100.0,  # Skip transient period
    )

    # Extract all outputs
    outputs = extractor.extract_all(
        output_types=[OutputType.TRANSCRIPTOME, OutputType.EXCHANGE_FLUXES],
    )
    print()
