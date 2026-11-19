# ImplicitRDP deployment

From the `i2rt` repository root, in the `ForceGello` environment:

```bash
python examples/implicitrdp_policy/run.py \
  --checkpoint ../ImplicitRDP/data/outputs/<run>/checkpoints/latest.ckpt \
  --validate-only
```

Remove `--validate-only` after checking the checkpoint. The yellow button
toggles policy control; Ctrl+C stops the robot. The script expects the
ImplicitRDP repository beside `i2rt`; use `--implicitrdp-root` if it is
elsewhere. It accepts only a checkpoint trained with the ForceGello absolute
7D joint command task: 20 Hz joint observations and inference, 5 Hz delayed
camera observations, and 60 Hz interpolated commands.
