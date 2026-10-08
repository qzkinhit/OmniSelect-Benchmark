# DaISy process-identification data

`cstr.dat.gz` (DaISy 98-002, continuous stirred tank reactor, 7,500 steps, the forecasting track
reads column 2, the concentration) and `steamgen.dat.gz` (DaISy 98-003, Abbott steam generator,
column 5, the drum pressure). `destill.dat.gz` is kept for completeness and is not read. The
tracks decompress `<name>.dat` next to the archive on first use and check its sha256
(docs/dataset_provenance.md). The uncompressed copies are ignored by Git.

Source: DaISy, KU Leuven ESAT-SISTA identification database. Its page asks users to cite the database
(see THIRD_PARTY_NOTICES.md).
