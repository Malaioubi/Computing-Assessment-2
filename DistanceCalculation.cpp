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

void runCalculations(const std::vector<std::pair<double, double>>& locations, const std::string& baseFilename, bool runInParallel) {
    auto start = std::chrono::high_resolution_clock::now();
    calculateDistances(locations, baseFilename, runInParallel);
    auto mid = std::chrono::high_resolution_clock::now();
    calculateWraparoundDistances(locations, baseFilename, runInParallel);
    auto end = std::chrono::high_resolution_clock::now();

    double runtimeStandard = std::chrono::duration_cast<std::chrono::milliseconds>(mid - start).count();
    double runtimeWraparound = std::chrono::duration_cast<std::chrono::milliseconds>(end - mid).count();

    std::cout << (runInParallel ? "Parallel" : "Serial") << " Execution Times:\n";
    std::cout << "  Standard Geometry Runtime: " << runtimeStandard << " ms\n";
    std::cout << "  Wraparound Geometry Runtime: " << runtimeWraparound << " ms\n";
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
    std::cout << "Enter base output filename : ";
    std::cin >> baseFilename;

    // User input for serial or parallel execution
    bool runInParallel;
    int executionChoice;
    std::cout << "Code execution method:\n1. Serial\n2. Parallel\nEnter choice (1 or 2): ";
    std::cin >> executionChoice;
    runInParallel = (executionChoice == 2);

    if (runInParallel) {
        // Parallel execution

        // User input for number of threads 
        int numThreads;
        std::cout << "Enter number of threads: ";
        std::cin >> numThreads;
        omp_set_num_threads(numThreads);

        // User input for scheduling type
        std::string scheduleType;
        std::cout << "Enter OpenMP scheduling type (static, dynamic): ";
        std::cin >> scheduleType;

        std::cout << "Running in parallel mode with " << numThreads << " threads...\n";
    } else {
        // Serial execution
        std::cout << "Running in serial mode...\n";
    }
    
    // User input for naive or optimised implementation
    int algorithmChoice;
    std::cout << "Select algorithm:\n1. Naive\n2. Optimised\nEnter choice (1 or 2): ";
    std::cin >> algorithmChoice;

    if (algorithmChoice == 1) {
        // Run naive calculations
        std::cout << "Running naive algorithm...\n";
        runCalculations(locations, baseFilename, runInParallel);
    } else if (algorithmChoice == 2) {
        // Run optimised calculations
        std::cout << "Running optimised algorithm...\n";
        calculateOptimisedDistances(locations, baseFilename, runInParallel);
    } else {
        std::cerr << "Invalid algorithm choice. Exiting program.\n";
        return 1;
    }

    return 0;
}

