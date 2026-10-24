# Source and artifact notices

The `framework` runtime is derived from YOLOv11-RGBT and Ultralytics and retains
its GNU Affero General Public License version 3 notices. Upstream sources:

- https://github.com/wandahangFY/YOLOv11-RGBT
- https://github.com/ultralytics/ultralytics

The SPLA kit was supplied for this research project. Its original archive hash
is recorded in `artifacts/records/input_manifest.json`. The distributed combined
runtime and integration are provided under the accompanying AGPL-3.0 terms.
Existing third-party source notices remain in the vendored files.

Publication modifications include ASCII normalization of comments, documentation,
and display strings, portable configuration aliases, English documentation,
tensor-only inference model exports, and a portable bundle loader. The original
runtime's source manifest and the publication's own content manifest are distinct:
normalizing source text necessarily changes file hashes.

Model hashes identify the original completed source checkpoints and the exported
tensor files separately. Original server pickle objects, local paths, raw dataset
material, temporary runs, and deployment credentials are not redistributed.
Prediction files are generated model outputs; the official dataset must be
obtained independently to reproduce their scores.
