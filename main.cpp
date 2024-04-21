#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <omp.h>
#include <cmath>

int main(int argc, char *argv[]) {
    unsigned int threads = std::stoi(argv[1]);
    if (threads == 0) threads = omp_get_max_threads();
    omp_set_num_threads(threads);
    std::string file_name = argv[2];
    std::string output_file_name = argv[3];
    double k = std::stod(argv[4]);
    unsigned char header[2];
    size_t width, height, max_v;
    std::ifstream in(file_name, std::ios::binary);
    std::ofstream out(output_file_name, std::ios::binary);
    if (out.fail()) {
        std::cout << "Permission denied or wrong path";
        exit(1);
    }
    if (in.fail()) {
        std::cout << "File not found";
        exit(1);
    }
    in >> header[0] >> header[1];
    out << header[0] << header[1] << '\n';
    if (header[0] != 'P' || (header[1] != '5' && header[1] != '6')) {
        std::cout << "Invalid file" << '\n';
        exit(1);
    }
    in >> width >> height >> max_v;
    out << width << ' ' << height << '\n' << max_v << '\n';
    size_t size = width * height;
    size_t size_channel = size;
    if (header[1] == '6') size *= 3;
    std::vector<uint8_t> arr(size);
    in.get();
    in.read(reinterpret_cast<char *>(&arr[0]), size);
    in.close();

    double time = omp_get_wtime();

    const int channels = header[1] == '5' ? 1 : 3;

    std::vector<std::vector<unsigned int>> values_br_result(channels, std::vector<unsigned int>(256));
#pragma omp parallel shared(values_br_result, arr, size, cout, channels) default(none)
    {
        unsigned int arr_thread[3][256];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 256; j++) {
                arr_thread[i][j] = 0;
            }
        }
#pragma omp for schedule(static)
        for (int i = 0; i < size; i += channels) {
            arr_thread[0][arr[i]]++;
            if (channels == 3) {
                arr_thread[1][arr[i + 1]]++;
                arr_thread[2][arr[i + 2]]++;
            }
        }
#pragma omp critical
        {
            for (int p = 0; p < values_br_result.size(); p++)
                for (int k = 0; k < 256; k++)
                    values_br_result[p][k] += arr_thread[p][k];
        }
    }
    unsigned char min_value = 255;
    unsigned char max_value = 0;
    int up_bound = size_channel * (1 - k);
    int low_bound = k * size_channel;
    for (int i = 0; i < values_br_result.size(); i++) {
        int sum = 0;
        for (int j = 0; j < 256; j++) {
            sum += values_br_result[i][j];
            if (sum > low_bound) {
                min_value = std::min(min_value, (unsigned char) j);
            }
            if (sum < up_bound) {
                max_value = std::max(max_value, (unsigned char) j);
            }
        }
    }
    max_value++;
    std::vector<unsigned char> result_values(256);
    for (int i = 0; i < 256; i++) {
        int mor = round(255.0 * (i - min_value) / (max_value - min_value));
        result_values[i] = mor < 0 ? 0 : (mor > 255 ? 255 : mor);
    }

#pragma omp parallel for shared(arr, result_values) default(none) schedule(dynamic, 65536)
    for (int i = 0; i < arr.size(); i++) arr[i] = result_values[arr[i]];

    std::cout << "Time(" << threads << " thread(s)): " << (omp_get_wtime() - time) * 1000 << " ms" << std::endl;

    out.write(reinterpret_cast<char *>(&arr[0]), size);
    out.close();
}
