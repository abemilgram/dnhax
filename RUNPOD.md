# Runpod queue worker

Deploy branch `macoslive`, Dockerfile `Dockerfile`, endpoint type Queue.
The container starts `handler.py`, which registers the Runpod handler.
Smoke-test input: `{"input":{"kind":"sample"}}`.

The handler runs the existing processor synchronously and returns the local job
ID and newly published scene manifests. Supported kinds are `sample`,
`reconstruct`, `joint`, `pair`, and `register`; `payload` uses the same fields as
the local processor. Invalid input or processing failures fail the Runpod job.

This image supports sample geometry and registration. Real reconstruction also
requires installing CUDA PyTorch, the selected model requirements, and the
checkpoint as described in README.md. Model weights are not bundled.

The existing application is a local SQLite/filesystem workspace. Capture and
scene IDs must already exist in the worker's SIMV1_DATA directory; submitting an
ID from a laptop does not upload its video. Set SIMV1_DATA to a mounted persistent
workspace when retaining data and use a single worker. Do not run the local
worker concurrently against that workspace. This adapter does not run the live
batch scheduler or migrate the frontend to Runpod's queue API.

Artifact paths in scene manifests are relative to the existing application's
`/api/artifacts/` route; the queue container does not host that HTTP route.
Retrieve artifacts from the workspace or serve them through the existing API.
