#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <limits>
#include <utility>
#include <chrono>
#include <random>
#include <omp.h>

// Function to read CSV file and store coordinates in a vector of pairs
std::vector<std::pair<double, double>> readCSV(const std::string& filename) {
    std::vector<std::pair<double, double>> locations;
    std::ifstream file(filename);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return locations;
    }

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string x_str, y_str;
        if (std::getline(ss, x_str, ',') && std::getline(ss, y_str)) {
            try {
                double x = std::stod(x_str);
                double y = std::stod(y_str);
                locations.emplace_back(x, y);
            } catch (const std::invalid_argument& e) {
                std::cerr << "Error: Invalid coordinate in line: " << line << std::endl;
            } catch (const std::out_of_range& e) {
                std::cerr << "Error: Coordinate out of range in line: " << line << std::endl;
            }
        } else {
            std::cerr << "Error: Malformed line: " << line << std::endl;
        }
    }

    file.close();
    return locations;
}

// Function to generate random locations 
std::vector<std::pair<double, double>> generateRandomLocations(const size_t numPoints) {
    std::vector<std::pair<double, double>> locations;
    locations.reserve(numPoints);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (size_t i = 0; i < numPoints; ++i) {
        double x = dis(gen);
        double y = dis(gen);
        locations.emplace_back(x, y);
    }

    return locations;
}

// Function to calculate distances for standard geometry either in serial or parallel
void calculateDistances(const std::vector<std::pair<double, double>>& locations, const std::string& baseFilename, bool runInParallel) {
    const size_t numPoints = locations.size();

    std::ofstream nearestStandardFile(baseFilename + "_standard_nearest.txt");
    std::ofstream furthestStandardFile(baseFilename + "_standard_furthest.txt");

    if (!nearestStandardFile.is_open() || !furthestStandardFile.is_open()) {
        std::cerr << "Error: Could not open output files.\n";
        return;
    }

    double totalNearestStandard = 0.0, totalFurthestStandard = 0.0;

    if (runInParallel) {
        // Parallel implementation
        #pragma omp parallel for schedule(runtime) reduction(+:totalNearestStandard, totalFurthestStandard)
        for (size_t i = 0; i < numPoints; ++i) {
            double nearestStandard = std::numeric_limits<double>::max();
            double furthestStandard = 0.0;

            for (size_t j = 0; j < numPoints; ++j) {
                if (i == j) continue;
                double dx = std::abs(locations[j].first - locations[i].first);
                double dy = std::abs(locations[j].second - locations[i].second);
                double dist = std::sqrt(dx * dx + dy * dy);

                nearestStandard = std::min(nearestStandard, dist);
                furthestStandard = std::max(furthestStandard, dist);
            }

            #pragma omp critical
            {
                nearestStandardFile << nearestStandard << "\n";
                furthestStandardFile << furthestStandard << "\n";
            }

            totalNearestStandard += nearestStandard;
            totalFurthestStandard += furthestStandard;
        }
    } else {
        // Serial implementation
        for (size_t i = 0; i < numPoints; ++i) {
            double nearestStandard = std::numeric_limits<double>::max();
            double furthestStandard = 0.0;

            for (size_t j = 0; j < numPoints; ++j) {
                if (i == j) continue;
                double dx = std::abs(locations[j].first - locations[i].first);
                double dy = std::abs(locations[j].second - locations[i].second);
                double dist = std::sqrt(dx * dx + dy * dy);

                nearestStandard = std::min(nearestStandard, dist);
                furthestStandard = std::max(furthestStandard, dist);
            }

            nearestStandardFile << nearestStandard << "\n";
            furthestStandardFile << furthestStandard << "\n";

            totalNearestStandard += nearestStandard;
            totalFurthestStandard += furthestStandard;
        }
    }

    double avgNearestStandard = totalNearestStandard / numPoints;
    double avgFurthestStandard = totalFurthestStandard / numPoints;

    std::cout << "Standard Geometry:\n";
    std::cout << "  Average Nearest Distance: " << avgNearestStandard << "\n";
    std::cout << "  Average Furthest Distance: " << avgFurthestStandard << "\n";

    nearestStandardFile.close();
    furthestStandardFile.close();
}

// Function to calculate distances for wraparound geometry either in serial or parallel
void calculateWraparoundDistances(const std::vector<std::pair<double, double>>& locations, const std::string& baseFilename, bool runInParallel) {
    const size_t numPoints = locations.size();

    std::ofstream nearestWraparoundFile(baseFilename + "_wraparound_nearest.txt");
    std::ofstream furthestWraparoundFile(baseFilename + "_wraparound_furthest.txt");

    if (!nearestWraparoundFile.is_open() || !furthestWraparoundFile.is_open()) {
        std::cerr << "Error: Could not open output files.\n";
        return;
    }

    double totalNearestWraparound = 0.0, totalFurthestWraparound = 0.0;

    if (runInParallel) {
        // Parallel implementation
        #pragma omp parallel for schedule(runtime) reduction(+:totalNearestWraparound, totalFurthestWraparound)
        for (size_t i = 0; i < numPoints; ++i) {
            double nearestWraparound = std::numeric_limits<double>::max();
            double furthestWraparound = 0.0;

            for (size_t j = 0; j < numPoints; ++j) {
                if (i == j) continue;
                double dx = std::abs(locations[j].first - locations[i].first);
                double dy = std::abs(locations[j].second - locations[i].second);
                double wrapDx = std::min(dx, 1.0 - dx);
                double wrapDy = std::min(dy, 1.0 - dy);
                double dist = std::sqrt(wrapDx * wrapDx + wrapDy * wrapDy);

                nearestWraparound = std::min(nearestWraparound, dist);
                furthestWraparound = std::max(furthestWraparound, dist);
            }

            #pragma omp critical
            {
                nearestWraparoundFile << nearestWraparound << "\n";
                furthestWraparoundFile << furthestWraparound << "\n";
            }

            totalNearestWraparound += nearestWraparound;
            totalFurthestWraparound += furthestWraparound;
        }
    } else {
        // Serial implementation
        for (size_t i = 0; i < numPoints; ++i) {
            double nearestWraparound = std::numeric_limits<double>::max();
            double furthestWraparound = 0.0;

            for (size_t j = 0; j < numPoints; ++j) {
                if (i == j) continue;
                double dx = std::abs(locations[j].first - locations[i].first);
                double dy = std::abs(locations[j].second - locations[i].second);
                double wrapDx = std::min(dx, 1.0 - dx);
                double wrapDy = std::min(dy, 1.0 - dy);
                double dist = std::sqrt(wrapDx * wrapDx + wrapDy * wrapDy);

                nearestWraparound = std::min(nearestWraparound, dist);
                furthestWraparound = std::max(furthestWraparound, dist);
            }

            nearestWraparoundFile << nearestWraparound << "\n";
            furthestWraparoundFile << furthestWraparound << "\n";

            totalNearestWraparound += nearestWraparound;
            totalFurthestWraparound += furthestWraparound;
        }
    }

    double avgNearestWraparound = totalNearestWraparound / numPoints;
    double avgFurthestWraparound = totalFurthestWraparound / numPoints;

    std::cout << "Wraparound Geometry:\n";
    std::cout << "  Average Nearest Distance: " << avgNearestWraparound << "\n";
    std::cout << "  Average Furthest Distance: " << avgFurthestWraparound << "\n";

    nearestWraparoundFile.close();
    furthestWraparoundFile.close();
}

// Optimised function to calculate distances for standard geometry either in serial or parallel
void calculateOptimisedDistances(const std::vector<std::pair<double, double>>& locations, const std::string& baseFilename, bool runInParallel) {
    const size_t numPoints = locations.size();
    std::vector<double> nearestDistances(numPoints, std::numeric_limits<double>::max());
    std::vector<double> furthestDistances(numPoints, 0.0);

    if (runInParallel) {
        // Parallel implementation with thread-local storage
        std::vector<std::vector<double>> threadNearestDistances, threadFurthestDistances;

        #pragma omp parallel
        {
            // Thread-local storage for nearest and furthest distances
            std::vector<double> localNearestDistances(numPoints, std::numeric_limits<double>::max());
            std::vector<double> localFurthestDistances(numPoints, 0.0);

            #pragma omp for schedule(dynamic)
            for (size_t i = 0; i < numPoints; ++i) {
                for (size_t j = i + 1; j < numPoints; ++j) { // Only process each pair once (i < j)
                    double dx = locations[j].first - locations[i].first;
                    double dy = locations[j].second - locations[i].second;
                    double dist = std::sqrt(dx * dx + dy * dy);

                    // Update thread-local distances for both points
                    localNearestDistances[i] = std::min(localNearestDistances[i], dist);
                    localFurthestDistances[i] = std::max(localFurthestDistances[i], dist);

                    localNearestDistances[j] = std::min(localNearestDistances[j], dist);
                    localFurthestDistances[j] = std::max(localFurthestDistances[j], dist);
                }
            }

            // Collect thread-local results
            #pragma omp critical
            {
                threadNearestDistances.push_back(std::move(localNearestDistances));
                threadFurthestDistances.push_back(std::move(localFurthestDistances));
            }
        }

        // Merge thread-local results into global results
        for (const auto& localNearest : threadNearestDistances) {
            for (size_t i = 0; i < numPoints; ++i) {
                nearestDistances[i] = std::min(nearestDistances[i], localNearest[i]);
            }
        }
        for (const auto& localFurthest : threadFurthestDistances) {
            for (size_t i = 0; i < numPoints; ++i) {
                furthestDistances[i] = std::max(furthestDistances[i], localFurthest[i]);
            }
        }

    } else {
        // Serial implementation
        for (size_t i = 0; i < numPoints; ++i) {
            double localNearest = std::numeric_limits<double>::max();
            double localFurthest = 0.0;

            for (size_t j = 0; j < numPoints; ++j) {
                if (i == j) continue;
                double dx = locations[j].first - locations[i].first;
                double dy = locations[j].second - locations[i].second;
                double dist = std::sqrt(dx * dx + dy * dy);

                localNearest = std::min(localNearest, dist);
                localFurthest = std::max(localFurthest, dist);
            }

            nearestDistances[i] = localNearest;
            furthestDistances[i] = localFurthest;
        }
    }

    // Calculate averages
    double totalNearest = 0.0, totalFurthest = 0.0;
    for (size_t i = 0; i < numPoints; ++i) {
        totalNearest += nearestDistances[i];
        totalFurthest += furthestDistances[i];
    }

    double avgNearest = totalNearest / numPoints;
    double avgFurthest = totalFurthest / numPoints;

    std::cout << "Optimised " << (runInParallel ? "Parallel" : "Serial") << " Standard Geometry:\n";
    std::cout << "  Average Nearest Distance: " << avgNearest << "\n";
    std::cout << "  Average Furthest Distance: " << avgFurthest << "\n";

    // Write results to files
    std::ofstream nearestFile(baseFilename + "_optimised_standard_nearest.txt");
    std::ofstream furthestFile(baseFilename + "_optimised_standard_furthest.txt");

    for (size_t i = 0; i < numPoints; ++i) {
        nearestFile << nearestDistances[i] << "\n";
        furthestFile << furthestDistances[i] << "\n";
    }

    nearestFile.close();
    furthestFile.close();
}

// Optimised function to calculate distances for wraparound geometry either in serial or parallel
void calculateOptimisedWraparoundDistances(const std::vector<std::pair<double, double>>& locations, const std::string& baseFilename, bool runInParallel) {
    const size_t numPoints = locations.size();
    std::vector<double> nearestDistances(numPoints, std::numeric_limits<double>::max());
    std::vector<double> furthestDistances(numPoints, 0.0);

    if (runInParallel) {
        // Parallel implementation with thread-local storage
        std::vector<std::vector<double>> threadNearestDistances, threadFurthestDistances;

        #pragma omp parallel
        {
            // Thread-local storage for nearest and furthest distances
            std::vector<double> localNearestDistances(numPoints, std::numeric_limits<double>::max());
            std::vector<double> localFurthestDistances(numPoints, 0.0);

            #pragma omp for schedule(dynamic)
            for (size_t i = 0; i < numPoints; ++i) {
                for (size_t j = i + 1; j < numPoints; ++j) { // Only process each pair once (i < j)
                    double dx = locations[j].first - locations[i].first;
                    double dy = locations[j].second - locations[i].second;
                    double wrapDx = std::min(dx, 1.0 - dx);
                    double wrapDy = std::min(dy, 1.0 - dy);
                    double dist = std::sqrt(wrapDx * wrapDx + wrapDy * wrapDy);

                    // Update thread-local distances for both points
                    localNearestDistances[i] = std::min(localNearestDistances[i], dist);
                    localFurthestDistances[i] = std::max(localFurthestDistances[i], dist);

                    localNearestDistances[j] = std::min(localNearestDistances[j], dist);
                    localFurthestDistances[j] = std::max(localFurthestDistances[j], dist);
                }
            }

            // Collect thread-local results
            #pragma omp critical
            {
                threadNearestDistances.push_back(std::move(localNearestDistances));
                threadFurthestDistances.push_back(std::move(localFurthestDistances));
            }
        }

        // Merge thread-local results into global results
        for (const auto& localNearest : threadNearestDistances) {
            for (size_t i = 0; i < numPoints; ++i) {
                nearestDistances[i] = std::min(nearestDistances[i], localNearest[i]);
            }
        }
        for (const auto& localFurthest : threadFurthestDistances) {
            for (size_t i = 0; i < numPoints; ++i) {
                furthestDistances[i] = std::max(furthestDistances[i], localFurthest[i]);
            }
        }

    } else {
        // Serial implementation
        for (size_t i = 0; i < numPoints; ++i) {
            double localNearest = std::numeric_limits<double>::max();
            double localFurthest = 0.0;

            for (size_t j = 0; j < numPoints; ++j) {
                if (i == j) continue;
                double dx = locations[j].first - locations[i].first;
                double dy = locations[j].second - locations[i].second;
                double wrapDx = std::min(dx, 1.0 - dx);
                double wrapDy = std::min(dy, 1.0 - dy);
                double dist = std::sqrt(wrapDx * wrapDx + wrapDy * wrapDy);

                localNearest = std::min(localNearest, dist);
                localFurthest = std::max(localFurthest, dist);
            }

            #pragma omp critical
            nearestDistances[i] = localNearest;
            furthestDistances[i] = localFurthest;
        }
    }

    // Calculate averages
    double totalNearest = 0.0, totalFurthest = 0.0;
    for (size_t i = 0; i < numPoints; ++i) {
        totalNearest += nearestDistances[i];
        totalFurthest += furthestDistances[i];
    }

    double avgNearest = totalNearest / numPoints;
    double avgFurthest = totalFurthest / numPoints;

    std::cout << "Optimised Wraparound " << (runInParallel ? "Parallel" : "Serial") << " Geometry:\n";
    std::cout << "  Average Nearest Distance: " << avgNearest << "\n";
    std::cout << "  Average Furthest Distance: " << avgFurthest << "\n";

    // Write results to files
    std::ofstream nearestFile(baseFilename + "_optimised_wraparound_nearest.txt");
    std::ofstream furthestFile(baseFilename + "_optimised_wraparound_furthest.txt");

    for (size_t i = 0; i < numPoints; ++i) {
        nearestFile << nearestDistances[i] << "\n";
        furthestFile << furthestDistances[i] << "\n";
    }

    nearestFile.close();
    furthestFile.close();
}

void runCalculations(const std::vector<std::pair<double, double>>& locations, const std::string& baseFilename, bool runInParallel, bool useOptimised) {
        
        if (useOptimised) {
        // Run the optimised calculations
        auto startOptimised = std::chrono::high_resolution_clock::now();
        calculateOptimisedDistances(locations, baseFilename + "_optimised", runInParallel);
        auto midOptimised = std::chrono::high_resolution_clock::now();
        calculateOptimisedWraparoundDistances(locations, baseFilename + "_optimised", runInParallel);
        auto endOptimised = std::chrono::high_resolution_clock::now();

        double runtimeOptimisedStandard = std::chrono::duration_cast<std::chrono::milliseconds>(midOptimised - startOptimised).count();
        double runtimeOptimisedWraparound = std::chrono::duration_cast<std::chrono::milliseconds>(endOptimised - midOptimised).count();

        std::cout << (runInParallel ? "Parallel" : "Serial") << " Optimised Execution Times:\n";
        std::cout << "  Standard Geometry Runtime: " << runtimeOptimisedStandard << " ms\n";
        std::cout << "  Wraparound Geometry Runtime: " << runtimeOptimisedWraparound << " ms\n";
    } else {
        // Run the naive calculations for both standard and wraparound geometries
        auto startNaive = std::chrono::high_resolution_clock::now();
        calculateDistances(locations, baseFilename + "_naive_standard", runInParallel);
        auto midNaive = std::chrono::high_resolution_clock::now();
        calculateWraparoundDistances(locations, baseFilename + "_naive_wraparound", runInParallel);
        auto endNaive = std::chrono::high_resolution_clock::now();

        double runtimeNaiveStandard = std::chrono::duration_cast<std::chrono::milliseconds>(midNaive - startNaive).count();
        double runtimeNaiveWraparound = std::chrono::duration_cast<std::chrono::milliseconds>(endNaive - midNaive).count();

        std::cout << (runInParallel ? "Parallel" : "Serial") << " Naive Execution Times:\n";
        std::cout << "  Standard Geometry Runtime: " << runtimeNaiveStandard << " ms\n";
        std::cout << "  Wraparound Geometry Runtime: " << runtimeNaiveWraparound << " ms\n";
    }
}


// Main function to run the program
int main() {
    std::vector<std::pair<double, double>> locations;
    int choice;
    size_t numPoints;

    // User input for method of location generation
    std::cout << "Select input method:\n1. Read from CSV file\n2. Generate random locations\nEnter choice (1 or 2): ";
    std::cin >> choice;

    if (choice == 1) {
        std::string filename;
        std::cout << "Enter CSV filename: ";
        std::cin >> filename;
        locations = readCSV(filename);

        if (locations.empty()) {
            std::cerr << "No locations loaded. Exiting program.\n";
            return 1;
        }
    } else if (choice == 2) {
        std::cout << "Enter number of random points to generate: ";
        std::cin >> numPoints;
        locations = generateRandomLocations(numPoints);
    } else {
        std::cerr << "Invalid choice. Exiting program.\n";
        return 1;
    }

    // User input for output filename
    std::string baseFilename;
    std::cout << "Enter base output filename: ";
    std::cin >> baseFilename;

    // User input for serial or parallel execution
    bool runInParallel;
    int executionChoice;
    std::cout << "Code execution method:\n1. Serial\n2. Parallel\nEnter choice (1 or 2): ";
    std::cin >> executionChoice;
    runInParallel = (executionChoice == 2);

    if (runInParallel) {
        // Parallel execution settings
        int numThreads;
        std::cout << "Enter number of threads: ";
        std::cin >> numThreads;
        omp_set_num_threads(numThreads);

        std::string scheduleType;
        std::cout << "Enter OpenMP scheduling type (static, dynamic): ";
        std::cin >> scheduleType;

        std::cout << "Running in parallel mode with " << numThreads << " threads using " << scheduleType << " scheduling.\n";
    } else {
        std::cout << "Running in serial mode...\n";
    }

    // User input for naive or optimised implementation
    int algorithmChoice;
    std::cout << "Select algorithm:\n1. Naive\n2. Optimised\nEnter choice (1 or 2): ";
    std::cin >> algorithmChoice;

    // Run calculations based on user choices
    if (algorithmChoice == 1) {
        std::cout << "Running naive algorithm...\n";
        runCalculations(locations, baseFilename, runInParallel, false);
    } else if (algorithmChoice == 2) {
        std::cout << "Running optimised algorithm...\n";
        runCalculations(locations, baseFilename, runInParallel, true);
    } else {
        std::cerr << "Invalid algorithm choice. Exiting program.\n";
        return 1;
    }
    
    return 0;
}
  
