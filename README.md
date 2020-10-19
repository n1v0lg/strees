# STrees

A prototype implementation of the C4.5 algorithm for secure decision tree training in MP-SPDZ.

Protocols etc. here: [Secure training of decision trees with continuous attributes](https://eprint.iacr.org/2020/1130).

## Disclaimer

This is a research prototype. No claims as to security or correctness. Please do not use in produciton.

## Setting up, testing, running

Requires Docker to run.

Requires MP-SPDZ repo (tested with commit [`3faa8a0`](https://github.com/data61/MP-SPDZ/commit/3faa8a0d4ab2a49eb94355ad8ff5c14f58cfb142)) to be in the parent directory. If you've cloned this project into `foo/bar/`, MP-SPDZ needs to be under `foo/bar` (or just adjust the shell scripts accordingly). Note that MP-SPDZ only needs to be there, doesn't need to be compiled or even compilable in your system as everything will run in a Docker container.

To set up, run (make sure Docker is running!):

```bash
bash build.sh
```

This might take a while. Once it's done you should get a bash terminal inside a Docker container.

The folder that contains `strees` is mounted into the container. This means that you can modify the source files *outside* Docker and the changes will show up inside the instance. 

From the Docker terminal prompt `cd ../strees/`.

Run `bash test.sh`. This should eventually print `All tests OK.`

To run all tests and examples with the output shown, run `bash run.sh test_all.py`. 

The core implementation is under `c45/c45.py`. A good entry point is the `c45` method. 

Note that once you exit the terminal, the Docker container will shut down. You don't need to rebuild it. Just run to restart it and get another bash session:

```bash
docker start mpspdz && docker exec -ti mpspdz /bin/bash
```

## Benchmarking

You can find the benchmark code under `/benchmarks`. See `/benchmarks/README.md` for more info. 

## Limitations

This is very much work in progress. Some limitations include:

* Discrete attributes are not yet supported. Only continuous attributes for now.
* There is only a fully oblivious version of the algorithm implemented, I don't take advantage of revealing some of the tree structure etc. yet.
* The algorithm runs exactly for the specified number of iterations.
* The continuous attribute values are currently just integer values.
* The number of samples must be a power of 2, as required by the underlying oblivious shuffle. Same goes for the number of attributes.

## Misc

*Representing classes.* There are different ways in which we can represent the class of a sample. The current implementation only deals with binary classes so a single binary column is sufficient. If we have more than two classes, one obvious representation is to assign an integer to each class and have a non-binary class column. However, this forces us to run additional equality checks. We can avoid these by using a separate binary indicator column for each class instead.
 
