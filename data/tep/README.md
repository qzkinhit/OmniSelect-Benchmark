# Tennessee Eastman Process

Braatz simulation files `d00.dat` to `d21.dat` (training) and `d00_te.dat` to `d21_te.dat`
(test), 52 process variables per row. `d00.dat` is stored transposed (52 x 500) and is read as
500 x 52. Test fault files contain the fault from sample 161, so the process track keeps rows
160 onward. Checksums of all 44 files are in `SHA256SUMS.txt` in this directory.

Source: the Braatz distribution (github.com/camaramm/tennessee-eastman-profBraatz), whose license
carries source and binary notice and non-endorsement conditions. The lineage of this copy is
recorded in docs/dataset_provenance.md and THIRD_PARTY_NOTICES.md.
