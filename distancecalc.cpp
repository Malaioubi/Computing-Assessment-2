#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <limits>
#include <utility>

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

// Function to calculate Euclidean distance
double calculateDistance(double x1, double y1, double x2, double y2) {
    return std::sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
}

// Function to calculate nearest and furthest distances
void calculateDistances(const std::vector<std::pair<double, double>>& locations) {
    std::ofstream nearestFile("nearest.txt");
    std::ofstream furthestFile("furthest.txt");

    if (!nearestFile.is_open() || !furthestFile.is_open()) {
        std::cerr << "Error: Could not open output files." << std::endl;
        return;
    }

    for (const auto& point : locations) {
        double nearest = std::numeric_limits<double>::max(); // Start with the largest possible value
        double furthest = 0.0; // Start with the smallest possible value

        for (const auto& otherPoint : locations) {
            if (point == otherPoint) continue; // Skip the same point

            double distance = calculateDistance(point.first, point.second, otherPoint.first, otherPoint.second);
            nearest = std::min(nearest, distance);
            furthest = std::max(furthest, distance);
        }

        nearestFile << nearest << "\n";
        furthestFile << furthest << "\n";
    }

    nearestFile.close();
    furthestFile.close();
    std::cout << "Nearest and furthest distances have been saved to 'nearest.txt' and 'furthest.txt'." << std::endl;
}

int main() {
    // Update the filename if needed
    std::string filename = "100000 locations.csv";
    std::vector<std::pair<double, double>> locations = readCSV(filename);

    // Verify that locations were loaded
    if (locations.empty()) {
        std::cerr << "No locations loaded. Exiting program." << std::endl;
        return 1;
    }

    // Calculate distances
    calculateDistances(locations);

    return 0;
}

