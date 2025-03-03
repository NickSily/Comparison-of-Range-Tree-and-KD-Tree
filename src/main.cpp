#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <memory>
#include <algorithm>
#include <iomanip>
#include <functional>
#include <cmath>

// Include KDTree and RangeTree headers
#include "../KdTree/KDTree.h"
#include "../RangeTree/RangeTree.h"

// Timer utility for benchmarking
class Timer
{
private:
    std::chrono::high_resolution_clock::time_point start_time;

public:
    Timer() : start_time(std::chrono::high_resolution_clock::now()) {}

    double elapsed() const
    {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(now - start_time).count();
    }

    void reset()
    {
        start_time = std::chrono::high_resolution_clock::now();
    }
};

// Represents a distribution type for point generation
enum class DistributionType
{
    UNIFORM_RANDOM,
    GAUSSIAN,
    GRID,
    CIRCLE,
    SKEWED
};

// Function to generate points according to different distributions
std::vector<std::vector<double>> generatePoints(
    size_t count, size_t dimensions, DistributionType dist_type, unsigned int seed = 42)
{

    std::vector<std::vector<double>> points;
    std::mt19937 gen(seed);

    switch (dist_type)
    {
    case DistributionType::UNIFORM_RANDOM:
    {
        // Uniform random distribution in [0, 1000]
        std::uniform_real_distribution<double> dist(0.0, 1000.0);

        for (size_t i = 0; i < count; ++i)
        {
            std::vector<double> point(dimensions);
            for (size_t j = 0; j < dimensions; ++j)
            {
                point[j] = dist(gen);
            }
            points.push_back(point);
        }
        break;
    }

    case DistributionType::GAUSSIAN:
    {
        // Gaussian distribution centered at 500 with std dev 100
        std::normal_distribution<double> dist(500.0, 100.0);

        for (size_t i = 0; i < count; ++i)
        {
            std::vector<double> point(dimensions);
            for (size_t j = 0; j < dimensions; ++j)
            {
                point[j] = dist(gen);
            }
            points.push_back(point);
        }
        break;
    }

    case DistributionType::GRID:
    {
        // Grid distribution
        size_t grid_size = static_cast<size_t>(std::ceil(std::pow(count, 1.0 / dimensions)));
        double step = 1000.0 / grid_size;

        std::vector<size_t> indices(dimensions, 0);
        size_t created = 0;

        // Generate points in a grid pattern
        while (created < count && indices[0] < grid_size)
        {
            std::vector<double> point(dimensions);
            for (size_t j = 0; j < dimensions; ++j)
            {
                point[j] = indices[j] * step;
            }
            points.push_back(point);

            // Increment indices (like counting in base grid_size)
            size_t dim = dimensions - 1;
            while (dim < dimensions)
            {
                indices[dim]++;
                if (indices[dim] < grid_size || dim == 0)
                    break;
                indices[dim] = 0;
                dim--;
            }

            created++;
        }
        break;
    }

    case DistributionType::CIRCLE:
    {
        // Points on a circle (only works meaningfully for 2D)
        if (dimensions >= 2)
        {
            double radius = 500.0;
            double center_x = 500.0;
            double center_y = 500.0;

            for (size_t i = 0; i < count; ++i)
            {
                std::vector<double> point(dimensions, 0.0);
                double angle = 2.0 * M_PI * i / count;
                point[0] = center_x + radius * std::cos(angle);
                point[1] = center_y + radius * std::sin(angle);

                // For higher dimensions, fill with random values
                for (size_t j = 2; j < dimensions; ++j)
                {
                    std::uniform_real_distribution<double> dist(0.0, 1000.0);
                    point[j] = dist(gen);
                }

                points.push_back(point);
            }
        }
        else
        {
            // Fallback to uniform for 1D
            std::uniform_real_distribution<double> dist(0.0, 1000.0);
            for (size_t i = 0; i < count; ++i)
            {
                std::vector<double> point(dimensions);
                for (size_t j = 0; j < dimensions; ++j)
                {
                    point[j] = dist(gen);
                }
                points.push_back(point);
            }
        }
        break;
    }

    case DistributionType::SKEWED:
    {
        // Skewed distribution: 80% of points in a small region
        size_t cluster_count = static_cast<size_t>(0.8 * count);
        size_t remainder = count - cluster_count;

        // Clustered region is [0, 200] (small corner)
        std::uniform_real_distribution<double> cluster_dist(0.0, 200.0);
        for (size_t i = 0; i < cluster_count; ++i)
        {
            std::vector<double> point(dimensions);
            for (size_t j = 0; j < dimensions; ++j)
            {
                point[j] = cluster_dist(gen);
            }
            points.push_back(point);
        }

        // Remainder spread out over [0, 1000]
        std::uniform_real_distribution<double> remainder_dist(0.0, 1000.0);
        for (size_t i = 0; i < remainder; ++i)
        {
            std::vector<double> point(dimensions);
            for (size_t j = 0; j < dimensions; ++j)
            {
                point[j] = remainder_dist(gen);
            }
            points.push_back(point);
        }

        // Shuffle to mix clustered and remainder points
        std::shuffle(points.begin(), points.end(), gen);
        break;
    }
    }

    return points;
}

// Function to generate query ranges of different sizes
std::vector<std::pair<std::vector<double>, std::vector<double>>> generateRangeQueries(
    size_t dimensions, size_t num_queries = 100)
{

    std::vector<std::pair<std::vector<double>, std::vector<double>>> queries;
    std::mt19937 gen(42);

    // Small ranges (about 1% of the space)
    for (size_t i = 0; i < num_queries / 3; ++i)
    {
        std::uniform_real_distribution<double> dist(0.0, 990.0);
        double x = dist(gen);
        double y = dimensions >= 2 ? dist(gen) : 0.0;

        std::vector<double> low(dimensions);
        std::vector<double> high(dimensions);

        low[0] = x;
        high[0] = x + 10.0;

        if (dimensions >= 2)
        {
            low[1] = y;
            high[1] = y + 10.0;
        }

        for (size_t j = 2; j < dimensions; ++j)
        {
            low[j] = dist(gen);
            high[j] = low[j] + 10.0;
        }

        queries.push_back({low, high});
    }

    // Medium ranges (about 10% of the space)
    for (size_t i = 0; i < num_queries / 3; ++i)
    {
        std::uniform_real_distribution<double> dist(0.0, 900.0);
        double x = dist(gen);
        double y = dimensions >= 2 ? dist(gen) : 0.0;

        std::vector<double> low(dimensions);
        std::vector<double> high(dimensions);

        low[0] = x;
        high[0] = x + 100.0;

        if (dimensions >= 2)
        {
            low[1] = y;
            high[1] = y + 100.0;
        }

        for (size_t j = 2; j < dimensions; ++j)
        {
            low[j] = dist(gen);
            high[j] = low[j] + 100.0;
        }

        queries.push_back({low, high});
    }

    // Large ranges (about 50% of the space)
    for (size_t i = 0; i < num_queries / 3 + num_queries % 3; ++i)
    {
        std::uniform_real_distribution<double> dist(0.0, 500.0);
        double x = dist(gen);
        double y = dimensions >= 2 ? dist(gen) : 0.0;

        std::vector<double> low(dimensions);
        std::vector<double> high(dimensions);

        low[0] = x;
        high[0] = x + 500.0;

        if (dimensions >= 2)
        {
            low[1] = y;
            high[1] = y + 500.0;
        }

        for (size_t j = 2; j < dimensions; ++j)
        {
            low[j] = dist(gen);
            high[j] = low[j] + 500.0;
        }

        queries.push_back({low, high});
    }

    return queries;
}

// Function to generate point queries (some existing, some not)
std::vector<std::vector<double>> generatePointQueries(
    const std::vector<std::vector<double>> &existing_points,
    size_t num_existing, size_t num_nonexisting)
{

    std::vector<std::vector<double>> queries;
    std::mt19937 gen(42);

    // Add some existing points
    std::uniform_int_distribution<size_t> idx_dist(0, existing_points.size() - 1);
    for (size_t i = 0; i < num_existing; ++i)
    {
        size_t idx = idx_dist(gen);
        queries.push_back(existing_points[idx]);
    }

    // Add some non-existing points
    size_t dimensions = existing_points[0].size();
    std::uniform_real_distribution<double> coord_dist(0.0, 1000.0);

    for (size_t i = 0; i < num_nonexisting; ++i)
    {
        std::vector<double> point(dimensions);
        for (size_t j = 0; j < dimensions; ++j)
        {
            point[j] = coord_dist(gen);
        }

        // Check if point exists (naive approach, but good enough for testing)
        bool exists = false;
        for (const auto &existing : existing_points)
        {
            bool match = true;
            for (size_t j = 0; j < dimensions; ++j)
            {
                if (std::abs(existing[j] - point[j]) > 1e-10)
                {
                    match = false;
                    break;
                }
            }
            if (match)
            {
                exists = true;
                break;
            }
        }

        if (!exists)
        {
            queries.push_back(point);
        }
        else
        {
            // Try again
            i--;
        }
    }

    return queries;
}

// Struct to hold benchmark results
struct BenchmarkResult
{
    std::string tree_type;
    std::string distribution;
    size_t num_points;
    size_t dimensions;
    double construction_time;
    double point_search_time_existing;
    double point_search_time_nonexisting;
    double small_range_query_time;
    double medium_range_query_time;
    double large_range_query_time;
    size_t small_range_avg_results;
    size_t medium_range_avg_results;
    size_t large_range_avg_results;
    // Only for KD Tree
    double nearest_neighbor_time;
};

// Function to conduct benchmark and return results
template <typename T, size_t D>
BenchmarkResult benchmarkKDTree(
    const std::vector<std::vector<T>> &points,
    const std::vector<std::vector<T>> &point_queries_existing,
    const std::vector<std::vector<T>> &point_queries_nonexisting,
    const std::vector<std::pair<std::vector<T>, std::vector<T>>> &range_queries,
    DistributionType dist_type)
{

    BenchmarkResult result;
    result.tree_type = "KD Tree";
    result.num_points = points.size();
    result.dimensions = D;

    switch (dist_type)
    {
    case DistributionType::UNIFORM_RANDOM:
        result.distribution = "Uniform Random";
        break;
    case DistributionType::GAUSSIAN:
        result.distribution = "Gaussian";
        break;
    case DistributionType::GRID:
        result.distribution = "Grid";
        break;
    case DistributionType::CIRCLE:
        result.distribution = "Circle";
        break;
    case DistributionType::SKEWED:
        result.distribution = "Skewed";
        break;
    }

    // Create a vector of correctly-sized points for KD Tree
    std::vector<std::vector<T>> points_d;
    for (const auto &p : points)
    {
        std::vector<T> point_d(p.begin(), p.begin() + D);
        points_d.push_back(point_d);
    }

    // Benchmark construction time
    Timer timer;
    KDTree<T, D> tree(points_d);
    result.construction_time = timer.elapsed();

    // Benchmark point search (existing points)
    timer.reset();
    for (const auto &query : point_queries_existing)
    {
        std::vector<T> query_d(query.begin(), query.begin() + D);
        tree.search(query_d);
    }
    result.point_search_time_existing = timer.elapsed() / point_queries_existing.size();

    // Benchmark point search (non-existing points)
    timer.reset();
    for (const auto &query : point_queries_nonexisting)
    {
        std::vector<T> query_d(query.begin(), query.begin() + D);
        tree.search(query_d);
    }
    result.point_search_time_nonexisting = timer.elapsed() / point_queries_nonexisting.size();

    // Prepare range queries for different sizes
    size_t num_queries = range_queries.size();
    size_t small_range_count = num_queries / 3;
    size_t medium_range_count = num_queries / 3;
    size_t large_range_count = num_queries - small_range_count - medium_range_count;

    // Benchmark small range queries
    timer.reset();
    size_t total_results_small = 0;
    for (size_t i = 0; i < small_range_count; ++i)
    {
        std::vector<T> low_d(range_queries[i].first.begin(), range_queries[i].first.begin() + D);
        std::vector<T> high_d(range_queries[i].second.begin(), range_queries[i].second.begin() + D);
        auto results = tree.rangeSearch(low_d, high_d);
        total_results_small += results.size();
    }
    result.small_range_query_time = timer.elapsed() / small_range_count;
    result.small_range_avg_results = total_results_small / small_range_count;

    // Benchmark medium range queries
    timer.reset();
    size_t total_results_medium = 0;
    for (size_t i = small_range_count; i < small_range_count + medium_range_count; ++i)
    {
        std::vector<T> low_d(range_queries[i].first.begin(), range_queries[i].first.begin() + D);
        std::vector<T> high_d(range_queries[i].second.begin(), range_queries[i].second.begin() + D);
        auto results = tree.rangeSearch(low_d, high_d);
        total_results_medium += results.size();
    }
    result.medium_range_query_time = timer.elapsed() / medium_range_count;
    result.medium_range_avg_results = total_results_medium / medium_range_count;

    // Benchmark large range queries
    timer.reset();
    size_t total_results_large = 0;
    for (size_t i = small_range_count + medium_range_count; i < num_queries; ++i)
    {
        std::vector<T> low_d(range_queries[i].first.begin(), range_queries[i].first.begin() + D);
        std::vector<T> high_d(range_queries[i].second.begin(), range_queries[i].second.begin() + D);
        auto results = tree.rangeSearch(low_d, high_d);
        total_results_large += results.size();
    }
    result.large_range_query_time = timer.elapsed() / large_range_count;
    result.large_range_avg_results = total_results_large / large_range_count;

    // Benchmark nearest neighbor queries
    timer.reset();
    for (const auto &query : point_queries_nonexisting)
    {
        std::vector<T> query_d(query.begin(), query.begin() + D);
        tree.nearestNeighbor(query_d);
    }
    result.nearest_neighbor_time = timer.elapsed() / point_queries_nonexisting.size();

    return result;
}

template <typename T, size_t D>
BenchmarkResult benchmarkRangeTree(
    const std::vector<std::vector<T>> &points,
    const std::vector<std::vector<T>> &point_queries_existing,
    const std::vector<std::vector<T>> &point_queries_nonexisting,
    const std::vector<std::pair<std::vector<T>, std::vector<T>>> &range_queries,
    DistributionType dist_type)
{

    BenchmarkResult result;
    result.tree_type = "Range Tree";
    result.num_points = points.size();
    result.dimensions = D;

    switch (dist_type)
    {
    case DistributionType::UNIFORM_RANDOM:
        result.distribution = "Uniform Random";
        break;
    case DistributionType::GAUSSIAN:
        result.distribution = "Gaussian";
        break;
    case DistributionType::GRID:
        result.distribution = "Grid";
        break;
    case DistributionType::CIRCLE:
        result.distribution = "Circle";
        break;
    case DistributionType::SKEWED:
        result.distribution = "Skewed";
        break;
    }

    // Create a vector of correctly-sized points for Range Tree
    std::vector<std::vector<T>> points_d;
    for (const auto &p : points)
    {
        std::vector<T> point_d(p.begin(), p.begin() + D);
        points_d.push_back(point_d);
    }

    // Benchmark construction time
    Timer timer;
    RangeTree<T, D> tree(points_d);
    result.construction_time = timer.elapsed();

    // Benchmark point search (existing points)
    timer.reset();
    for (const auto &query : point_queries_existing)
    {
        std::vector<T> query_d(query.begin(), query.begin() + D);
        tree.search(query_d);
    }
    result.point_search_time_existing = timer.elapsed() / point_queries_existing.size();

    // Benchmark point search (non-existing points)
    timer.reset();
    for (const auto &query : point_queries_nonexisting)
    {
        std::vector<T> query_d(query.begin(), query.begin() + D);
        tree.search(query_d);
    }
    result.point_search_time_nonexisting = timer.elapsed() / point_queries_nonexisting.size();

    // Prepare range queries for different sizes
    size_t num_queries = range_queries.size();
    size_t small_range_count = num_queries / 3;
    size_t medium_range_count = num_queries / 3;
    size_t large_range_count = num_queries - small_range_count - medium_range_count;

    // Benchmark small range queries
    timer.reset();
    size_t total_results_small = 0;
    for (size_t i = 0; i < small_range_count; ++i)
    {
        std::vector<T> low_d(range_queries[i].first.begin(), range_queries[i].first.begin() + D);
        std::vector<T> high_d(range_queries[i].second.begin(), range_queries[i].second.begin() + D);
        auto results = tree.rangeSearch(low_d, high_d);
        total_results_small += results.size();
    }
    result.small_range_query_time = timer.elapsed() / small_range_count;
    result.small_range_avg_results = total_results_small / small_range_count;

    // Benchmark medium range queries
    timer.reset();
    size_t total_results_medium = 0;
    for (size_t i = small_range_count; i < small_range_count + medium_range_count; ++i)
    {
        std::vector<T> low_d(range_queries[i].first.begin(), range_queries[i].first.begin() + D);
        std::vector<T> high_d(range_queries[i].second.begin(), range_queries[i].second.begin() + D);
        auto results = tree.rangeSearch(low_d, high_d);
        total_results_medium += results.size();
    }
    result.medium_range_query_time = timer.elapsed() / medium_range_count;
    result.medium_range_avg_results = total_results_medium / medium_range_count;

    // Benchmark large range queries
    timer.reset();
    size_t total_results_large = 0;
    for (size_t i = small_range_count + medium_range_count; i < num_queries; ++i)
    {
        std::vector<T> low_d(range_queries[i].first.begin(), range_queries[i].first.begin() + D);
        std::vector<T> high_d(range_queries[i].second.begin(), range_queries[i].second.begin() + D);
        auto results = tree.rangeSearch(low_d, high_d);
        total_results_large += results.size();
    }
    result.large_range_query_time = timer.elapsed() / large_range_count;
    result.large_range_avg_results = total_results_large / large_range_count;

    // Range Tree doesn't have nearest neighbor, so set to 0
    result.nearest_neighbor_time = 0.0;

    return result;
}

// Function to write benchmark results to a file
void writeResultsToFile(
    const std::vector<BenchmarkResult> &results,
    const std::string &filename = "tree_benchmark_results.txt")
{

    std::ofstream file(filename);

    if (!file.is_open())
    {
        std::cerr << "Error opening output file: " << filename << std::endl;
        return;
    }

    // Write header
    file << std::setw(12) << "Tree Type" << ','
         << std::setw(16) << "Distribution" << ','
         << std::setw(12) << "Num Points" << ','
         << std::setw(10) << "Dimensions" << ','
         << std::setw(16) << "Construction (ms)" << ','
         << std::setw(16) << "Point Search (ms)" << ','
         << std::setw(20) << "Non-Exist Search (ms)" << ','
         << std::setw(20) << "Small Range Query (ms)" << ','
         << std::setw(20) << "Medium Range Query (ms)" << ','
         << std::setw(20) << "Large Range Query (ms)" << ','
         << std::setw(20) << "Small Range Results" << ','
         << std::setw(20) << "Medium Range Results" << ','
         << std::setw(20) << "Large Range Results" << ','
         << std::setw(20) << "Nearest Neighbor (ms)" << std::endl;

    // Write data rows
    for (const auto &result : results)
    {
        file << std::setw(12) << result.tree_type << ','
             << std::setw(16) << result.distribution << ','
             << std::setw(12) << result.num_points << ','
             << std::setw(10) << result.dimensions << ','
             << std::setw(16) << std::fixed << std::setprecision(4) << result.construction_time << ','
             << std::setw(16) << std::fixed << std::setprecision(4) << result.point_search_time_existing << ','
             << std::setw(20) << std::fixed << std::setprecision(4) << result.point_search_time_nonexisting << ','
             << std::setw(20) << std::fixed << std::setprecision(4) << result.small_range_query_time << ','
             << std::setw(20) << std::fixed << std::setprecision(4) << result.medium_range_query_time << ','
             << std::setw(20) << std::fixed << std::setprecision(4) << result.large_range_query_time << ','
             << std::setw(20) << result.small_range_avg_results << ','
             << std::setw(20) << result.medium_range_avg_results << ','
             << std::setw(20) << result.large_range_avg_results << ','
             << std::setw(20) << std::fixed << std::setprecision(4) << result.nearest_neighbor_time << std::endl;
    }

    file.close();

    std::cout << "Benchmark results written to: " << filename << std::endl;
}

// Write a summary analysis of the benchmarks
void writeAnalysisSummary(
    const std::vector<BenchmarkResult> &results,
    const std::string &filename = "tree_benchmark_analysis.txt")
{

    std::ofstream file(filename);

    if (!file.is_open())
    {
        std::cerr << "Error opening output file: " << filename << std::endl;
        return;
    }

    file << "KD Tree vs Range Tree Performance Analysis" << std::endl;
    file << "=========================================" << std::endl
         << std::endl;

    // Analyze construction time by dataset size
    file << "1. Construction Time Analysis by Dataset Size" << std::endl;
    file << "-----------------------------------------" << std::endl;

    // Group by distribution type, dimensions, and dataset size
    std::vector<std::string> distributions = {"Uniform Random", "Gaussian", "Grid", "Circle", "Skewed"};
    std::vector<size_t> dimensions = {2, 3};
    std::vector<size_t> dataset_sizes = {100, 1000, 10000, 100000};

    for (const auto &dist : distributions)
    {
        file << "\nDistribution: " << dist << std::endl;

        for (const auto &dim : dimensions)
        {
            file << "\n  Dimensions: " << dim << std::endl;
            file << "  ------------------------------" << std::endl;
            file << "  | Dataset Size | KD Tree (ms) | Range Tree (ms) | Ratio (Range/KD) |" << std::endl;
            file << "  |--------------|-------------|----------------|-----------------|" << std::endl;

            for (const auto &size : dataset_sizes)
            {
                double kd_time = 0, range_time = 0;
                bool kd_found = false, range_found = false;

                for (const auto &result : results)
                {
                    if (result.distribution == dist && result.dimensions == dim && result.num_points == size)
                    {
                        if (result.tree_type == "KD Tree")
                        {
                            kd_time = result.construction_time;
                            kd_found = true;
                        }
                        else if (result.tree_type == "Range Tree")
                        {
                            range_time = result.construction_time;
                            range_found = true;
                        }
                    }
                }

                if (kd_found && range_found)
                {
                    double ratio = range_time / kd_time;
                    file << "  | " << std::setw(12) << size
                         << " | " << std::setw(11) << std::fixed << std::setprecision(2) << kd_time
                         << " | " << std::setw(14) << std::fixed << std::setprecision(2) << range_time
                         << " | " << std::setw(15) << std::fixed << std::setprecision(2) << ratio << " |" << std::endl;
                }
            }
        }
    }

    // Analyze range query performance by range size
    file << "\n\n2. Range Query Performance Analysis" << std::endl;
    file << "--------------------------------" << std::endl;

    for (const auto &dist : distributions)
    {
        file << "\nDistribution: " << dist << std::endl;

        for (const auto &dim : dimensions)
        {
            file << "\n  Dimensions: " << dim << std::endl;
            file << "  Dataset Size: 10000 points" << std::endl;
            file << "  ----------------------------------------------------------------------" << std::endl;
            file << "  | Range Size | KD Tree (ms) | Range Tree (ms) | Ratio (Range/KD) | Avg Results |" << std::endl;
            file << "  |------------|-------------|----------------|-----------------|------------|" << std::endl;

            for (const auto &result : results)
            {
                if (result.distribution == dist && result.dimensions == dim && result.num_points == 10000)
                {
                    if (result.tree_type == "KD Tree")
                    {
                        // Find the corresponding Range Tree result
                        for (const auto &range_result : results)
                        {
                            if (range_result.tree_type == "Range Tree" &&
                                range_result.distribution == dist &&
                                range_result.dimensions == dim &&
                                range_result.num_points == 10000)
                            {

                                // Small ranges
                                double ratio_small = range_result.small_range_query_time / result.small_range_query_time;
                                file << "  | Small      | " << std::setw(11) << std::fixed << std::setprecision(2) << result.small_range_query_time
                                     << " | " << std::setw(14) << std::fixed << std::setprecision(2) << range_result.small_range_query_time
                                     << " | " << std::setw(15) << std::fixed << std::setprecision(2) << ratio_small
                                     << " | " << std::setw(10) << result.small_range_avg_results << " |" << std::endl;

                                // Medium ranges
                                double ratio_medium = range_result.medium_range_query_time / result.medium_range_query_time;
                                file << "  | Medium     | " << std::setw(11) << std::fixed << std::setprecision(2) << result.medium_range_query_time
                                     << " | " << std::setw(14) << std::fixed << std::setprecision(2) << range_result.medium_range_query_time
                                     << " | " << std::setw(15) << std::fixed << std::setprecision(2) << ratio_medium
                                     << " | " << std::setw(10) << result.medium_range_avg_results << " |" << std::endl;

                                // Large ranges
                                double ratio_large = range_result.large_range_query_time / result.large_range_query_time;
                                file << "  | Large      | " << std::setw(11) << std::fixed << std::setprecision(2) << result.large_range_query_time
                                     << " | " << std::setw(14) << std::fixed << std::setprecision(2) << range_result.large_range_query_time
                                     << " | " << std::setw(15) << std::fixed << std::setprecision(2) << ratio_large
                                     << " | " << std::setw(10) << result.large_range_avg_results << " |" << std::endl;

                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    // Point search performance
    file << "\n\n3. Point Search Performance Analysis" << std::endl;
    file << "---------------------------------" << std::endl;

    for (const auto &dist : distributions)
    {
        file << "\nDistribution: " << dist << std::endl;

        for (const auto &dim : dimensions)
        {
            file << "\n  Dimensions: " << dim << std::endl;
            file << "  Dataset Size: 10000 points" << std::endl;
            file << "  ---------------------------------------------------------------------" << std::endl;
            file << "  | Query Type | KD Tree (ms) | Range Tree (ms) | Ratio (Range/KD) |" << std::endl;
            file << "  |------------|-------------|----------------|-----------------|" << std::endl;

            for (const auto &result : results)
            {
                if (result.distribution == dist && result.dimensions == dim && result.num_points == 10000)
                {
                    if (result.tree_type == "KD Tree")
                    {
                        // Find the corresponding Range Tree result
                        for (const auto &range_result : results)
                        {
                            if (range_result.tree_type == "Range Tree" &&
                                range_result.distribution == dist &&
                                range_result.dimensions == dim &&
                                range_result.num_points == 10000)
                            {

                                // Existing points
                                double ratio_existing = range_result.point_search_time_existing / result.point_search_time_existing;
                                file << "  | Existing   | " << std::setw(11) << std::fixed << std::setprecision(2) << result.point_search_time_existing
                                     << " | " << std::setw(14) << std::fixed << std::setprecision(2) << range_result.point_search_time_existing
                                     << " | " << std::setw(15) << std::fixed << std::setprecision(2) << ratio_existing << " |" << std::endl;

                                // Non-existing points
                                double ratio_nonexisting = range_result.point_search_time_nonexisting / result.point_search_time_nonexisting;
                                file << "  | Non-exist  | " << std::setw(11) << std::fixed << std::setprecision(2) << result.point_search_time_nonexisting
                                     << " | " << std::setw(14) << std::fixed << std::setprecision(2) << range_result.point_search_time_nonexisting
                                     << " | " << std::setw(15) << std::fixed << std::setprecision(2) << ratio_nonexisting << " |" << std::endl;

                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    // Summary and conclusions
    file << "\n\n4. Overall Analysis and Conclusions" << std::endl;
    file << "--------------------------------" << std::endl;

    // Calculate average speedups/slowdowns across all tests
    double avg_construction_ratio = 0.0;
    double avg_point_search_ratio = 0.0;
    double avg_range_query_ratio = 0.0;
    int construction_count = 0;
    int point_search_count = 0;
    int range_query_count = 0;

    for (size_t i = 0; i < results.size(); i += 2)
    {
        if (i + 1 < results.size())
        {
            const auto &kd_result = results[i].tree_type == "KD Tree" ? results[i] : results[i + 1];
            const auto &range_result = results[i].tree_type == "Range Tree" ? results[i] : results[i + 1];

            // Construction time ratio
            avg_construction_ratio += range_result.construction_time / kd_result.construction_time;
            construction_count++;

            // Point search ratio (average of existing and non-existing)
            double point_ratio = (range_result.point_search_time_existing / kd_result.point_search_time_existing +
                                  range_result.point_search_time_nonexisting / kd_result.point_search_time_nonexisting) /
                                 2.0;
            avg_point_search_ratio += point_ratio;
            point_search_count++;

            // Range query ratio (average of small, medium, large)
            double range_ratio = (range_result.small_range_query_time / kd_result.small_range_query_time +
                                  range_result.medium_range_query_time / kd_result.medium_range_query_time +
                                  range_result.large_range_query_time / kd_result.large_range_query_time) /
                                 3.0;
            avg_range_query_ratio += range_ratio;
            range_query_count++;
        }
    }

    if (construction_count > 0)
        avg_construction_ratio /= construction_count;
    if (point_search_count > 0)
        avg_point_search_ratio /= point_search_count;
    if (range_query_count > 0)
        avg_range_query_ratio /= range_query_count;

    file << "Average performance ratios (Range Tree / KD Tree):" << std::endl;
    file << "- Construction time: " << std::fixed << std::setprecision(2) << avg_construction_ratio
         << "x (Range Tree is " << (avg_construction_ratio > 1 ? "slower" : "faster") << ")" << std::endl;
    file << "- Point search time: " << std::fixed << std::setprecision(2) << avg_point_search_ratio
         << "x (Range Tree is " << (avg_point_search_ratio > 1 ? "slower" : "faster") << ")" << std::endl;
    file << "- Range query time: " << std::fixed << std::setprecision(2) << avg_range_query_ratio
         << "x (Range Tree is " << (avg_range_query_ratio > 1 ? "slower" : "faster") << ")" << std::endl;

    file << "\nStrengths and Weaknesses:" << std::endl;

    file << "\nKD Tree:" << std::endl;
    file << "- Strengths: ";
    if (avg_construction_ratio > 1)
        file << "Faster construction, ";
    if (avg_point_search_ratio > 1)
        file << "Faster point search, ";
    file << "Supports nearest neighbor queries" << std::endl;
    file << "- Weaknesses: ";
    if (avg_range_query_ratio < 1)
        file << "Slower range queries, ";
    file << "Less efficient for higher dimensions" << std::endl;

    file << "\nRange Tree:" << std::endl;
    file << "- Strengths: ";
    if (avg_construction_ratio < 1)
        file << "Faster construction, ";
    if (avg_point_search_ratio < 1)
        file << "Faster point search, ";
    if (avg_range_query_ratio < 1)
        file << "Faster range queries, ";
    file << "Optimized for orthogonal range queries" << std::endl;
    file << "- Weaknesses: ";
    if (avg_construction_ratio > 1)
        file << "Slower construction, ";
    if (avg_point_search_ratio > 1)
        file << "Slower point search, ";
    if (avg_range_query_ratio > 1)
        file << "Slower range queries, ";
    file << "No built-in nearest neighbor support, ";
    file << "Higher memory usage" << std::endl;

    file << "\nRecommendations:" << std::endl;
    file << "- Use KD Tree when: ";
    if (avg_construction_ratio > 1)
        file << "Construction speed is important, ";
    if (avg_point_search_ratio > 1)
        file << "Point searches are common, ";
    file << "Nearest neighbor queries are needed, ";
    file << "Memory usage is a concern" << std::endl;

    file << "- Use Range Tree when: ";
    if (avg_range_query_ratio < 1)
        file << "Range queries are the primary operation, ";
    file << "Query performance is more important than construction time or memory usage" << std::endl;

    file << "\nNote: This analysis is based on the specific implementations being tested and may vary with different optimizations." << std::endl;

    file.close();

    std::cout << "Analysis summary written to: " << filename << std::endl;
}

int main()
{
    std::cout << "Starting KD Tree vs Range Tree benchmarking..." << std::endl;

    std::vector<BenchmarkResult> all_results;

    // Define dataset sizes to test
    std::vector<size_t> dataset_sizes = {100, 1000, 10000};

    // For very large datasets, only test uniform distribution to save time
    std::vector<size_t> large_dataset_sizes = {100000};

    // Define distributions to test
    std::vector<DistributionType> distributions = {
        DistributionType::UNIFORM_RANDOM,
        DistributionType::GAUSSIAN,
        DistributionType::GRID,
        DistributionType::CIRCLE,
        DistributionType::SKEWED};

    // Define dimensions to test
    std::vector<size_t> dimensions = {2, 3};

    // Benchmark both tree types with various distributions and sizes
    for (const auto &dim : dimensions)
    {
        for (const auto &dist_type : distributions)
        {
            for (const auto &size : dataset_sizes)
            {
                std::cout << "Benchmarking with " << size << " points, ";

                switch (dist_type)
                {
                case DistributionType::UNIFORM_RANDOM:
                    std::cout << "uniform random distribution, ";
                    break;
                case DistributionType::GAUSSIAN:
                    std::cout << "gaussian distribution, ";
                    break;
                case DistributionType::GRID:
                    std::cout << "grid distribution, ";
                    break;
                case DistributionType::CIRCLE:
                    std::cout << "circle distribution, ";
                    break;
                case DistributionType::SKEWED:
                    std::cout << "skewed distribution, ";
                    break;
                }

                std::cout << dim << " dimensions..." << std::endl;

                // Generate points for this test
                auto points = generatePoints(size, std::max(dim, size_t(3)), dist_type);

                // Generate queries
                auto point_queries_existing = generatePointQueries(points, 50, 0);
                auto point_queries_nonexisting = generatePointQueries(points, 0, 50);
                auto range_queries = generateRangeQueries(std::max(dim, size_t(3)), 90);

                // Run benchmarks for both tree types
                if (dim == 2)
                {
                    auto kd_result = benchmarkKDTree<double, 2>(
                        points, point_queries_existing, point_queries_nonexisting, range_queries, dist_type);
                    all_results.push_back(kd_result);

                    auto range_result = benchmarkRangeTree<double, 2>(
                        points, point_queries_existing, point_queries_nonexisting, range_queries, dist_type);
                    all_results.push_back(range_result);
                }
                else if (dim == 3)
                {
                    auto kd_result = benchmarkKDTree<double, 3>(
                        points, point_queries_existing, point_queries_nonexisting, range_queries, dist_type);
                    all_results.push_back(kd_result);

                    auto range_result = benchmarkRangeTree<double, 3>(
                        points, point_queries_existing, point_queries_nonexisting, range_queries, dist_type);
                    all_results.push_back(range_result);
                }
            }
        }

        // Only benchmark uniform distribution for very large datasets
        for (const auto &size : large_dataset_sizes)
        {
            std::cout << "Benchmarking with " << size << " points, uniform random distribution, "
                      << dim << " dimensions..." << std::endl;

            auto dist_type = DistributionType::UNIFORM_RANDOM;
            auto points = generatePoints(size, std::max(dim, size_t(3)), dist_type);

            // Generate queries
            auto point_queries_existing = generatePointQueries(points, 50, 0);
            auto point_queries_nonexisting = generatePointQueries(points, 0, 50);
            auto range_queries = generateRangeQueries(std::max(dim, size_t(3)), 90);

            // Run benchmarks for both tree types
            if (dim == 2)
            {
                auto kd_result = benchmarkKDTree<double, 2>(
                    points, point_queries_existing, point_queries_nonexisting, range_queries, dist_type);
                all_results.push_back(kd_result);

                auto range_result = benchmarkRangeTree<double, 2>(
                    points, point_queries_existing, point_queries_nonexisting, range_queries, dist_type);
                all_results.push_back(range_result);
            }
            else if (dim == 3)
            {
                auto kd_result = benchmarkKDTree<double, 3>(
                    points, point_queries_existing, point_queries_nonexisting, range_queries, dist_type);
                all_results.push_back(kd_result);

                auto range_result = benchmarkRangeTree<double, 3>(
                    points, point_queries_existing, point_queries_nonexisting, range_queries, dist_type);
                all_results.push_back(range_result);
            }
        }
    }

    // Write results to files
    writeResultsToFile(all_results);
    writeAnalysisSummary(all_results);

    std::cout << "Benchmarking complete!" << std::endl;

    return 0;
}