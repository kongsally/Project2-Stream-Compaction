CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Sally Kong
* Tested on: Windows 8, i7-5500U CPU @ 2.40GHz 2.40 GHz, GEForce 920M (Personal)


**Summary:** This project is an implementation of a GPU stream compaction in CUDA,
from scratch. This is a widely used algorithm that I later plan to use to accelerate my path tracer.

A few different versions of the *Scan* (*Prefix Sum*)
algorithm were implemented: a CPU version, and a few GPU implementations: "naive" and
"work-efficient." 

**Algorithm overview & details:** There are two primary references for details
on the implementation of scan and stream compaction.

* The [slides on Parallel Algorithms](https://github.com/CIS565-Fall-2015/cis565-fall-2015.github.io/raw/master/lectures/2-Parallel-Algorithms.pptx)
  for Scan, Stream Compaction, and Work-Efficient Parallel Scan.
* GPU Gems 3, Chapter 39 - [Parallel Prefix Sum (Scan) with CUDA](http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html).

## Write-up

* Compare all of these GPU Scan implementations (Naive, Work-Efficient, and
  Thrust) to the serial CPU version of Scan. Plot a graph of the comparison
  (with array size on the independent axis).
  * You should use CUDA events for timing. Be sure **not** to include any
    explicit memory operations in your performance measurements, for
    comparability.
  * To guess at what might be happening inside the Thrust implementation, take
    a look at the Nsight timeline for its execution.

* Write a brief explanation of the phenomena you see here.
  * Can you find the performance bottlenecks? Is it memory I/O? Computation? Is
    it different for each implementation?

* Paste the output of the test program into a triple-backtick block in your
  README.
  * If you add your own tests (e.g. for radix sort or to test additional corner
    cases), be sure to mention it explicitly.

These questions should help guide you in performance analysis on future
assignments, as well.

