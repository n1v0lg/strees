# STrees Benchmarking

## Launching AWS instances

- From the AWS EC2 console, start `MP-SPDZ-0`, `MP-SPDZ-1`, and `MP-SPDZ-2`.
- For each instance, look up its public IP.
- Add the IPs to your `/etc/hosts` file (you will need `sudo` on this):

```
x.x.x.x   MP-SPDZ-0
x.x.x.x   MP-SPDZ-1
x.x.x.x   MP-SPDZ-2
``` 

where x.x.x.x is the public IP address of each instance. 

- Once that is done, you can ssh into each instance via:

```bash
ssh -i path_to_ssh_key ubuntu@MP-SPDZ-0
```

- Don't forget to set the right permissions on the key file:

```bash
chmod 400 path_to_key_file
```

## Running benchmarks

From each instance, you can use the `all_*.sh` scripts in the `/benchmarks` dir to run a benchmark.

Use `bash all_*.sh --help` to get a list of all command line args (sorry not done with that yet, so it currently won't work!).

For each benchmark, you first need to run a compile pass (note: this could take a long time). For example to run the micro benchmark script, use:

```bash
bash all_micro.sh --mode compile --pid ${PID}
```

Note that you don't need to change ${PID}, since each instances has its party ID set as an environment variable already (`echo ${PID}` to verify).

Once the compile pass finishes on each machine, run:

```bash
bash all_micro.sh --mode run --pid ${PID}
```

You should start the run on `MP-SPDZ-2` and `MP-SPDZ-1` first, and then get the timing from `MP-SPDZ-0`, since otherwise, the run time will include the time `MP-SPDZ-0` took to wait for the other instances to initiate a run.

Running the two above commands should have added entries to two output files, `timing-compile-micro.py.csv` and `timing-run-micro.py.csv`. An entry is a single row, for example:

```csv
micro.py,argmax-16,2.238
```

Which tells you the name of the high-level benchmark script you used (`micro.py`), and arguments to that script (`argmax-16` which means argmax on 16 values), and the time it took.

To get the running times for active security, use:

```bash
bash all_micro.sh --mode run --pid ${PID} --mal
```

Note that *don't* need to recompile!

If you need to see the output of your runs, use the debug flag:

```bash
bash all_micro.sh --mode run --pid ${PID} --debug
```

If you make local changes to the run scripts, you can use `push_to_remote_.sh` to scp the changes over to the AWS instances. 

__IMPORTANT__: the `push_to_remote_.sh` script will scp everything from your local benchmarks dir, so make sure to not have any local timing files; those will overwrite the files on the AWS machines.

## More details on benchmark pipeline

The overall setup for the benchmark scripts is that an `all_*.sh` will call `run.sh` with several different configurations, which in turn calls a `*.py` file.

The `run.sh` script takes a range of command line arguments and handles invoking calling an actual benchmark file, timing the run, and persisting the result. 

The `*.py` files contain the actual benchmarks implemented in MP-SPDZ.

## Benchmarks to run

We still need timings for:
 
 - `all_attrs.sh` (passive and active)
 - `all_etoe.sh` (passive and active)
 - `all_breakdown.sh` (active)
 - `all_micro.sh` (active)
 
 For each one, run in compile mode first (will take a long time!), then in run mode.

## Common Issues

- Compile stage raises `RuntimeError: maximum recursion depth exceeded`. For large programs, the optimization pass performs very deep recursion which can lead to this exception. To fix it, try increasing recursion limit in Python. In `MP-SPDZ/compile.py`, add `sys.setrecursionlimit(2000)`. Try increasing past `2000` if the issue persists.
