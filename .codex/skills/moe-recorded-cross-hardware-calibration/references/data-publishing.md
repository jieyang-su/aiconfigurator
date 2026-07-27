# Data Publishing

Publish only after the final default collector path and final-output gate pass.

## What To Commit

Project data:

- compact `*.txt` / `*.parquet` files used by AIC lookup;
- collector/materializer code changes required to reproduce them;
- small manifest or method files needed for future reproduction.

Method artifacts:

- frozen truth manifest;
- truth scripts;
- SGLang instrumentation reference patch;
- parser/gate/replay scripts;
- concise gate summary.

Do not commit:

- large nsys reports;
- raw profiler traces;
- intermediate collector run directories;
- temporary CSV exploration dumps;
- zombie debug logs;
- date-stamped one-off scripts when an archived no-date version exists.

## Default Verification Before PR

Run the default AIC command for the model and verify:

- total errors are acceptable or zero;
- expected compact files are present;
- no unexpected source/debug directories are emitted in no-keep mode;
- gate summary is generated from final compact files;
- missing truth points are documented.

The PR should explain:

- model case matrix;
- hardware data used;
- backend/SGLang runtime version and image used;
- truth source used;
- gate result;
- known gaps such as missing EP coverage.
