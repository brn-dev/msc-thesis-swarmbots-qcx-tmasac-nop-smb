# External benchmark experiments

Each scenario has its own directory and uses the fixed scenario configuration in
`scenario_experiment_common.py`. Run its `run.py` directly and optionally select
one of the eight policy variants:

```powershell
.\.venv\Scripts\python.exe experiments\external_benchmarks\vmas_balance\run.py tmasac_nop
.\.venv\Scripts\python.exe experiments\external_benchmarks\mamujoco_halfcheetah\run.py mat_qcx
```

The default variant is `tmasac`. Available variants are `mat_qcx`,
`mat_qcx_nop`, `mat_dec`, `mat_dec_nop`, `mat_ind`, `mat_ind_nop`, `tmasac`,
and `tmasac_nop`.

The MaMuJoCo set contains the standard homogeneous multi-agent partitions that
the shared MAT/TMASAC policy can represent without padding or ignored action
dimensions. The VMAS set is intentionally curated rather than exhaustive: it
covers cooperative manipulation, navigation, transport, exploration, coverage,
collective motion, traffic coordination, and competitive play.
