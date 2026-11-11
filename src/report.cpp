#include "report.hpp"

#include <fstream>
#include <iostream>
#include <stdexcept>

namespace report {

void print(const std::vector<optim::OptimizerResult>& results) {
    for (const auto& r : results) {
        std::cout << "optimizer: " << r.name << "\n"
                  << "  initial loss: " << r.initial_loss << "\n"
                  << "  final loss:   " << r.final_loss << "\n"
                  << "  updates:      " << r.updates << "\n"
                  << "  time (ms):    " << r.time_ms << "\n";

        if (r.epochs_to_converge < 0)
            std::cout << "  convergence:  not reached\n";
        else
            std::cout << "  convergence:  reached after " << r.epochs_to_converge << " epochs\n";
    }
}

void save_summary_csv(const std::vector<optim::OptimizerResult>& results,
                      const std::string& filename)
{
    std::ofstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("could not open " + filename);

    file << "optimizer,initial_loss,final_loss,epochs_to_converge,updates,time_ms,converged\n";
    for (const auto& r : results) {
        file << r.name << ","
             << r.initial_loss << ","
             << r.final_loss << ","
             << r.epochs_to_converge << ","
             << r.updates << ","
             << r.time_ms << ","
             << (r.epochs_to_converge >= 0 ? "yes" : "no") << "\n";
    }
}

void save_loss_history_csv(const std::vector<optim::OptimizerResult>& results,
                           const std::string& filename)
{
    std::ofstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("could not open " + filename);

    file << "optimizer,epoch,loss\n";
    for (const auto& r : results)
        for (const auto& pt : r.losses)
            file << r.name << "," << pt.x << "," << pt.y << "\n";
}

} // namespace report
