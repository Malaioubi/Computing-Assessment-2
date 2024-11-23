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
    std::mt19937 gen(rd()); // Mersenne Twister engine
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

// Function to calculate nearest and furthest distances using standard geometry with OpenMP
void calculateDistances(const std::vector<std::pair<double, double>>& locations) {
    std::ofstream nearestFile("nearest.txt");
    std::ofstream furthestFile("furthest.txt");

    if (!nearestFile.is_open() || !furthestFile.is_open()) {
        std::cerr << "Error: Could not open output files." << std::endl;
        return;
    }

    double totalNearest = 0.0;
    double totalFurthest = 0.0;

    #pragma omp parallel
    {
        double threadNearestSum = 0.0;
        double threadFurthestSum = 0.0;

        #pragma omp for schedule(dynamic)
        for (size_t i = 0; i < locations.size(); ++i) {
            double nearest = std::numeric_limits<double>::max();
            double furthest = 0.0;

            for (size_t j = 0; j < locations.size(); ++j) {
                if (i == j) continue;

                double distance = calculateDistance(locations[i].first, locations[i].second,
                                                    locations[j].first, locations[j].second);
                nearest = std::min(nearest, distance);
                furthest = std::max(furthest, distance);
            }

            threadNearestSum += nearest;
            threadFurthestSum += furthest;

            #pragma omp critical
            {
                nearestFile << nearest << "\n";
                furthestFile << furthest << "\n";
            }
        }

        #pragma omp atomic
        totalNearest += threadNearestSum;
        #pragma omp atomic
        totalFurthest += threadFurthestSum;
    }

    nearestFile.close();
    furthestFile.close();

    double avgNearest = totalNearest / locations.size();
    double avgFurthest = totalFurthest / locations.size();
    std::cout << "Average nearest distance (standard): " << avgNearest << std::endl;
    std::cout << "Average furthest distance (standard): " << avgFurthest << std::endl;
}

// Function to calculate nearest and furthest distances using wraparound geometry with OpenMP
void calculateWraparoundDistances(const std::vector<std::pair<double, double>>& locations) {
    std::ofstream nearestFile("nearest_wraparound.txt");
    std::ofstream furthestFile("furthest_wraparound.txt");

    if (!nearestFile.is_open() || !furthestFile.is_open()) {
        std::cerr << "Error: Could not open output files for wraparound distances." << std::endl;
        return;
    }

    double totalNearest = 0.0;
    double totalFurthest = 0.0;

    #pragma omp parallel
    {
        double threadNearestSum = 0.0;
        double threadFurthestSum = 0.0;

        #pragma omp for schedule(dynamic)
        for (size_t i = 0; i < locations.size(); ++i) {
            double nearest = std::numeric_limits<double>::max();
            double furthest = 0.0;

            for (size_t j = 0; j < locations.size(); ++j) {
                if (i == j) continue;

                double distance = calculateWraparoundDistance(locations[i].first, locations[i].second,
                                                              locations[j].first, locations[j].second);
                nearest = std::min(nearest, distance);
                furthest = std::max(furthest, distance);
            }

            threadNearestSum += nearest;
            threadFurthestSum += furthest;

            #pragma omp critical
            {
                nearestFile << nearest << "\n";
                furthestFile << furthest << "\n";
            }
        }

        #pragma omp atomic
        totalNearest += threadNearestSum;
        #pragma omp atomic
        totalFurthest += threadFurthestSum;
    }

    nearestFile.close();
    furthestFile.close();

    double avgNearest = totalNearest / locations.size();
    double avgFurthest = totalFurthest / locations.size();
    std::cout << "Average nearest distance (wraparound): " << avgNearest << std::endl;
    std::cout << "Average furthest distance (wraparound): " << avgFurthest << std::endl;
}

int main() {
    // Prompt user to choose input method
    std::cout << "Select input method:\n";
    std::cout << "1. Read from CSV file\n";
    std::cout << "2. Generate random locations\n";
    std::cout << "Enter choice (1 or 2): ";
    int choice;
    std::cin >> choice;

    std::vector<std::pair<double, double>> locations;

    if (choice == 1) {
        std::string filename;
        std::cout << "Enter CSV filename: ";
        std::cin >> filename;

        locations = readCSV(filename);
        if (locations.empty()) {
            std::cerr << "No locations loaded. Exiting program." << std::endl;
            return 1;
        }
    } else if (choice == 2) {
        size_t numPoints;
        std::cout << "Enter number of random points to generate: ";
        std::cin >> numPoints;

        locations = generateRandomLocations(numPoints);
    } else {
        std::cerr << "Invalid choice. Exiting program." << std::endl;
        return 1;
    }

    // Measure runtime for standard distances
    auto start = std::chrono::high_resolution_clock::now();
    calculateDistances(locations);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Standard geometry runtime: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms" << std::endl;

    // Measure runtime for wraparound distances
    start = std::chrono::high_resolution_clock::now();
    calculateWraparoundDistances(locations);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Wraparound geometry runtime: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms" << std::endl;

    return 0;
}
