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

// Function to calculate Euclidean distance
double calculateDistance(double x1, double y1, double x2, double y2) {
    return std::sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
}

// Function to calculate wraparound distance
double calculateWraparoundDistance(double x1, double y1, double x2, double y2) {
    double dx = std::abs(x2 - x1);
    double dy = std::abs(y2 - y1);

    double wrapDx = std::min(dx, 1.0 - dx);
    double wrapDy = std::min(dy, 1.0 - dy);

    return std::sqrt(wrapDx * wrapDx + wrapDy * wrapDy);
}

// Combined function to calculate distances and averages
void calculateAllDistances(const std::vector<std::pair<double, double>>& locations, const std::string& baseFilename) {
    size_t numPoints = locations.size();

    // Prepare files for output
    std::ofstream nearestStandardFile(baseFilename + "_standard_nearest.txt");
    std::ofstream furthestStandardFile(baseFilename + "_standard_furthest.txt");
    std::ofstream nearestWraparoundFile(baseFilename + "_wraparound_nearest.txt");
    std::ofstream furthestWraparoundFile(baseFilename + "_wraparound_furthest.txt");

    if (!nearestStandardFile.is_open() || !furthestStandardFile.is_open() || 
        !nearestWraparoundFile.is_open() || !furthestWraparoundFile.is_open()) {
        std::cerr << "Error: Could not open output files.\n";
        return;
    }

    // Variables for averages
    double totalNearestStandard = 0.0, totalFurthestStandard = 0.0;
    double totalNearestWraparound = 0.0, totalFurthestWraparound = 0.0;

    // Measure standard geometry runtime
    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for reduction(+:totalNearestStandard, totalFurthestStandard)
    for (size_t i = 0; i < numPoints; ++i) {
        double nearestStandard = std::numeric_limits<double>::max();
        double furthestStandard = 0.0;

        for (size_t j = 0; j < numPoints; ++j) {
            if (i == j) continue;
            double dist = calculateDistance(locations[i].first, locations[i].second, locations[j].first, locations[j].second);
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

    auto end = std::chrono::high_resolution_clock::now();
    double runtimeStandard = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Measure wraparound geometry runtime
    start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for reduction(+:totalNearestWraparound, totalFurthestWraparound)
    for (size_t i = 0; i < numPoints; ++i) {
        double nearestWraparound = std::numeric_limits<double>::max();
        double furthestWraparound = 0.0;

        for (size_t j = 0; j < numPoints; ++j) {
            if (i == j) continue;
            double dist = calculateWraparoundDistance(locations[i].first, locations[i].second, locations[j].first, locations[j].second);
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

    end = std::chrono::high_resolution_clock::now();
    double runtimeWraparound = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Calculate averages
    double avgNearestStandard = totalNearestStandard / numPoints;
    double avgFurthestStandard = totalFurthestStandard / numPoints;
    double avgNearestWraparound = totalNearestWraparound / numPoints;
    double avgFurthestWraparound = totalFurthestWraparound / numPoints;

    // Output results
    std::cout << "Standard Geometry:\n";
    std::cout << "  Runtime: " << runtimeStandard << " ms\n";
    std::cout << "  Average Nearest Distance: " << avgNearestStandard << "\n";
    std::cout << "  Average Furthest Distance: " << avgFurthestStandard << "\n";

    std::cout << "Wraparound Geometry:\n";
    std::cout << "  Runtime: " << runtimeWraparound << " ms\n";
    std::cout << "  Average Nearest Distance: " << avgNearestWraparound << "\n";
    std::cout << "  Average Furthest Distance: " << avgFurthestWraparound << "\n";

    // Close files
    nearestStandardFile.close();
    furthestStandardFile.close();
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

    // Output filename base
    std::string baseFilename;
    std::cout << "Enter base output filename (without extension): ";
    std::cin >> baseFilename;

    // Perform calculations
    calculateAllDistances(locations, baseFilename);

    return 0;
}
