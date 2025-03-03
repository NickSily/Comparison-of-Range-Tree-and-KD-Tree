# KD Tree vs Range Tree Benchmark

This benchmark provides a comprehensive comparison between KD Tree and Range Tree data structures across various datasets, sizes, and operations. The analysis helps determine which spatial data structure is more suitable for different scenarios and applications.

## Data Structures Overview

- **KD Tree**: A space-partitioning data structure that recursively partitions points in k-dimensional space. Each level of the tree divides the space along a specific dimension, cycling through dimensions as the tree grows deeper.

- **Range Tree**: A multi-dimensional spatial data structure optimized for orthogonal range queries. It consists of a nested hierarchy of binary search trees, with one tree per dimension.

## Benchmark Methodology

The benchmark evaluates performance across:

- **Dataset Sizes**: 100, 1,000, 10,000, and 100,000 points
- **Dimensions**: 2D and 3D spaces
- **Operations**:
  - Construction time
  - Point search (existing and non-existing points)
  - Range queries (small, medium, and large ranges)
  - Nearest neighbor search (KD Tree only)

## Data Distributions

The benchmark tests five different point distributions to simulate various real-world scenarios:

1. **Uniform Random Distribution**: Points are randomly distributed throughout the entire space with equal probability. This creates an unpredictable, evenly spread pattern that represents general-case scenarios with no specific structure.

2. **Gaussian Distribution**: Points are clustered around a central location (mean) with decreasing density as distance from center increases. This follows a normal distribution curve and simulates naturally occurring clusters found in many real-world datasets.

3. **Grid Distribution**: Points are arranged in a regular, evenly-spaced grid pattern. This highly structured arrangement tests performance with perfectly organized data and predictable spacing between points.

4. **Circle Distribution**: Points are arranged along the circumference of a circle (in 2D) or a spherical surface (conceptually, in higher dimensions). This represents a convex arrangement where points form a boundary around empty space.

5. **Skewed Distribution**: 80% of points are concentrated in a small region of the space, with the remaining 20% scattered throughout. This tests performance with highly imbalanced data distributions that contain dense clusters and sparse regions.

## Key Findings

### Construction Time

- **KD Tree** generally has faster construction time, especially for larger datasets.
- **Range Tree** construction scales less favorably with dataset size due to its more complex nested structure.
- The construction time gap widens as the number of dimensions increases.

### Search Operations

- **Point Search**: KD Tree typically outperforms Range Tree for exact point lookups, with the advantage increasing in higher dimensions.
- **Range Queries**: Range Tree excels at orthogonal range queries, particularly for medium to large range sizes, which is its primary design purpose.
- **Small Range Queries**: Both structures perform similarly for small range queries with few results.
- **Large Range Queries**: Range Tree maintains better performance as the query size increases.

### Distribution Effects

- **Uniform & Grid Distributions**: Both trees show consistent performance.
- **Gaussian & Skewed Distributions**: Range Tree handles clustered data more efficiently for range queries.
- **Circle Distribution**: KD Tree adapts better to this convex arrangement for point searches.

### Dimensionality

- Performance gap between the two structures widens with increasing dimensions.
- Range Tree's advantage for range queries becomes more pronounced in 3D compared to 2D.

## Recommendations

- **Use KD Tree when**:
  - Construction speed is important
  - Point searches and nearest neighbor queries are the primary operations
  - Memory usage is a concern
  - Working primarily with low-dimensional data

- **Use Range Tree when**:
  - Range queries are the dominant operation
  - Query performance is more important than construction time
  - Working with data that has clustering tendencies
  - Memory usage is not a significant constraint

## How to Run the Benchmark

1. Compile the benchmark code:
   ```
   g++ -o tree_benchmark tree-comparison-benchmark.cpp -std=c++17 -O3
   ```

2. Run the benchmark:
   ```
   ./tree_benchmark
   ```

3. Analyze the results in the generated files:
   - `tree_benchmark_results.txt`: Raw benchmark data
   - `tree_benchmark_analysis.txt`: Detailed analysis and recommendations

## Files Description

- `KdTree.h`: Implementation of the KD Tree data structure
- `RangeTree.h`: Implementation of the Range Tree data structure
- `tree-comparison-benchmark.cpp`: The benchmark code
- `tree_benchmark_results.txt`: Generated CSV-style results
- `tree_benchmark_analysis.txt`: Comprehensive analysis report

## Conclusion

The choice between KD Tree and Range Tree depends on the specific use case, with KD Trees offering better general-purpose performance and Range Trees specializing in efficient range queries. This benchmark provides data-driven guidance to help make this selection based on your application's needs.


##  bin
Machine Code files

## src
Source files ( .cpp and .h or .hpp)

## test-unit
Cache Test Module for testing. Test early and often kids!

## Makefile
To run Program Type "make" in Linux terminal
To run test type "make test" in Terminal
