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
            double x = std::stod(x_str);
            double y = std::stod(y_str);
            locations.emplace_back(x, y);
        }
    }

    file.close();
    return locations;
}

// Function to generate random locations within [0, 1] x [0, 1]
std::vector<std::pair<double, double>> generateRandomLocations(size_t numPoints) {
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

void calculateDistances(const std::vector<std::pair<double, double>>& locations, const std::string& baseFilename, bool runInParallel) {
    size_t numPoints = locations.size();

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

void calculateWraparoundDistances(const std::vector<std::pair<double, double>>& locations, const std::string& baseFilename, bool runInParallel) {
    size_t numPoints = locations.size();

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

    // User input for serial or parallel execution
    bool runInParallel;
    std::cout << "Do you want to run the code in parallel? (1 for Yes, 0 for No): ";
    std::cin >> runInParallel;

    // Output filename base
    std::string baseFilename;
    std::cout << "Enter base output filename (without extension): ";
    std::cin >> baseFilename;

    if (runInParallel) {
        // Parallel execution
        std::string scheduleType;
        std::cout << "Enter OpenMP scheduling type (static, dynamic): ";
        std::cin >> scheduleType;

        // Parallel calculations for standard and wraparound geometries
        calculateDistances(locations, baseFilename, true);
        calculateWraparoundDistances(locations, baseFilename, true);
    } else {
        // Serial execution
        std::cout << "Running in serial mode...\n";

        auto start = std::chrono::high_resolution_clock::now();
        calculateDistances(locations, baseFilename, false); // Serial for standard geometry
        auto mid = std::chrono::high_resolution_clock::now();
        calculateWraparoundDistances(locations, baseFilename, false); // Serial for wraparound geometry
        auto end = std::chrono::high_resolution_clock::now();

        double runtimeStandard = std::chrono::duration_cast<std::chrono::milliseconds>(mid - start).count();
        double runtimeWraparound = std::chrono::duration_cast<std::chrono::milliseconds>(end - mid).count();

        // Output runtimes for clarity
        std::cout << "Serial Execution Times:\n";
        std::cout << "  Standard Geometry Runtime: " << runtimeStandard << " ms\n";
        std::cout << "  Wraparound Geometry Runtime: " << runtimeWraparound << " ms\n";
    }

    return 0;
}

